import argparse
import fsspec
from fv3net.pipelines.common import list_timesteps
import logging
import xarray as xr
import vcm

from itertools import product
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions

from . import funcs

CATEGORIES = (
    "fv_srf_wnd_coarse.res",
    "sfc_data_coarse",
    "fv_tracer_coarse.res",
    "fv_core_coarse.res",
)

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("url", help="root directory of time steps")
    parser.add_argument("output", help="Location of output zarr")
    parser.add_argument("-s", "--n-steps", default=-1, type=int)
    parser.add_argument("--no-init", action="store_true")
    args, pipeline_args = parser.parse_known_args()

    times = list_timesteps(args.url)
    if args.n_steps != -1:
        times = times[: args.n_steps]

    fs = fsspec.filesystem("gs")
    categories = CATEGORIES
    tiles = [1, 2, 3, 4, 5, 6]

    # get schema for first timestep
    if not args.no_init:
        logging.info("Getting the schema")
        time: str = times[0]
        schema = xr.merge(
            [
                funcs.get_schema(fs, funcs._file(args.url, time, category, tile=1))
                for category in categories
            ]
        )
        print("Schema:")
        print(schema)

        logging.info("Initializing output zarr")
        store = funcs._get_store(args.output)
        output_m = vcm.ZarrMapping.from_schema(
            store, schema, dims=["time", "tile"], coords={"tile": tiles, "time": times}
        )

    items = product(times, categories, tiles)
    with beam.Pipeline(options=PipelineOptions(pipeline_args)) as p:

        (
            p
            | beam.Create(items)
            | "OpenFiles" >> beam.ParDo(funcs.get_timestep, fs, args.url)
            | "Insert" >> beam.Map(funcs.insert_timestep, args.output)
        )

    logging.info("Job completed succesfully!")
