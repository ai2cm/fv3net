import argparse
import logging
import os
from itertools import product
from typing import Tuple

import apache_beam as beam
import fsspec
import xarray as xr
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.utils import retry

import vcm
from fv3net.pipelines.common import list_timesteps

CATEGORIES = (
    "fv_srf_wnd_coarse.res",
    "sfc_data_coarse",
    "fv_tracer_coarse.res",
    "fv_core_coarse.res",
)


def _file(url: str, time: str, category: str, tile: int) -> str:
    return os.path.join(url, time, f"{time}.{category}.tile{tile}.nc")


@retry.with_exponential_backoff(num_retries=3)
def get_timestep(key, fs, url) -> xr.Dataset:
    time, category, tile = key
    location = _file(url, time, category, tile)
    logging.info(f"Opening {location}")
    with fs.open(location, "rb") as f:
        ds = xr.open_dataset(f).load()
    ds = vcm.standardize_metadata(ds)

    for variable in ds:
        yield key, ds[variable]


def insert_timestep(
    item: Tuple[Tuple[str, str, int], xr.DataArray], out_path: str,
):
    key, data = item
    time, category, tile = key
    # The ZarrMapping doesn't serialize so reinit it here.
    output = vcm.ZarrMapping(_get_store(out_path))
    name = data.name
    logging.info(f"Inserting {name} at {time} for {category} tile {tile}")
    output[time, tile] = data.to_dataset()


@retry.with_exponential_backoff(num_retries=3)
def get_schema(fs: fsspec.AbstractFileSystem, url: str) -> xr.Dataset:
    logging.info(f"Grabbing schema from {url}")
    with fs.open(url, "rb") as f:
        return vcm.standardize_metadata(xr.open_dataset(f))


def _get_store(output: str):
    return fsspec.get_mapper(output)


def main(argv):

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("url", help="root directory of time steps")
    parser.add_argument("output", help="Location of output zarr")
    parser.add_argument("-s", "--n-steps", default=-1, type=int)
    parser.add_argument("--no-init", action="store_true")
    args, pipeline_args = parser.parse_known_args(argv)

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
                get_schema(fs, _file(args.url, time, category, tile=1))
                for category in categories
            ]
        )
        print("Schema:")
        print(schema)

        logging.info("Initializing output zarr")
        store = _get_store(args.output)
        vcm.ZarrMapping.from_schema(
            store, schema, dims=["time", "tile"], coords={"tile": tiles, "time": times}
        )

    items = product(times, categories, tiles)
    with beam.Pipeline(options=PipelineOptions(pipeline_args)) as p:

        (
            p
            | beam.Create(items)
            | "OpenFiles" >> beam.ParDo(get_timestep, fs, args.url)
            | "Insert" >> beam.Map(insert_timestep, args.output)
        )

    logging.info("Job completed succesfully!")
