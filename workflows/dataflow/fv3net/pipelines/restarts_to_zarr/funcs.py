import argparse
from typing import Iterable, Tuple, List
import logging
import os
from itertools import product

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


Key = Tuple[str, str, int]


@retry.with_exponential_backoff(num_retries=3)
def get_timestep(
    key: Key, fs: fsspec.AbstractFileSystem, url: str, select_variables: List[str]
) -> Iterable[Tuple[Key, xr.DataArray]]:
    time, category, tile = key
    location = _file(url, time, category, tile)
    logging.info(f"Opening {location}")
    with fs.open(location, "rb") as f:
        ds = xr.open_dataset(f).load()
    ds = vcm.standardize_metadata(ds)
    if len(select_variables) > 0:
        ds = ds[select_variables]
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
        ds = vcm.standardize_metadata(xr.open_dataset(f).load())
    return ds


def _get_store(output: str):
    return fsspec.get_mapper(output)


def main(argv):

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("url", help="root directory of time steps")
    parser.add_argument("output", help="Location of output zarr")
    parser.add_argument(
        "--no-coarse-suffix",
        action="store_true",
        help="use if restart files do not have `_coarse` suffix in name",
    )
    parser.add_argument("-s", "--n-steps", default=-1, type=int)
    parser.add_argument("--no-init", action="store_true")
    parser.add_argument(
        "--select-categories",
        nargs="*",
        type=str,
        default=[],
        help="If provided, only open these categories of restarts to save resources.",
    )
    parser.add_argument(
        "--select-variables",
        nargs="*",
        type=str,
        default=[],
        help="If provided, save only these variables",
    )
    parser.add_argument(
        "--select-daily-times",
        nargs="*",
        type=str,
        default=[],
        help="If provided, save only these hours of day (HHMMSS format)",
    )
    args, pipeline_args = parser.parse_known_args(argv)

    times = list_timesteps(args.url)
    if args.n_steps != -1:
        times = times[: args.n_steps]
    if len(args.select_daily_times) > 0:
        times = [t for t in times if t[-6:] in args.select_daily_times]
    print("Saving times: ", times)

    fs = fsspec.filesystem("gs")
    categories = (
        args.select_categories if len(args.select_categories) > 0 else CATEGORIES
    )
    if args.no_coarse_suffix:
        categories = [category.replace("_coarse", "") for category in categories]

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
        ).drop_vars("tile", errors="ignore")
        print("Schema:")
        print(schema)

        logging.info("Initializing output zarr")
        store = _get_store(args.output)
        vcm.ZarrMapping.from_schema(
            store, schema, dims=["time", "tile"], coords={"tile": tiles, "time": times}
        )
        logging.info("Done initializing zarr.")

    items = product(times, categories, tiles)
    with beam.Pipeline(options=PipelineOptions(pipeline_args)) as p:

        (
            p
            | beam.Create(items)
            | "OpenFiles"
            >> beam.ParDo(get_timestep, fs, args.url, args.select_variables)
            | "Insert" >> beam.Map(insert_timestep, args.output)
        )

    logging.info("Job completed succesfully!")
