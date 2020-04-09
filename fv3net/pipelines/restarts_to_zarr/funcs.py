import logging
import os
from typing import Tuple

import fsspec
import xarray as xr
from apache_beam.utils import retry

import vcm


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
