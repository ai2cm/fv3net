import logging
import os
from itertools import product
from typing import Callable, Hashable, Iterable, Sequence, Tuple, TypeVar

import fsspec
import xarray as xr
import zarr
from distributed import Client
from toolz import curry

import fv3net
import vcm
from fv3net.pipelines import list_timesteps

logging.basicConfig(level=logging.INFO)

CATEGORIES=('fv_srf_wnd_coarse.res', 'sfc_data_coarse', 'fv_tracer_coarse.res', 'fv_core_coarse.res')


def _file(url: str, time: str, category: str, tile: int) -> str:
    return os.path.join(url, time, f"{time}.{category}.tile{tile}.nc")


@curry
def get_timestep(fs, url, time, category, tile) -> xr.Dataset:
    location = _file(url, time, category, tile)
    logging.info(f"Opening {location}")
    with fs.open(location, "rb") as f:
        ds = xr.open_dataset(f).load()
        return vcm.standardize_metadata(ds)


def insert_timestep(output: fv3net.ZarrMapping, get: Callable[[str, str, int], xr.Dataset], key: Tuple[str, str, int]):
    time, category, tile = key
    data = get(time, category, tile)
    logging.info(f"Setting data at {time} for {category} tile {tile}")
    output[time, tile] = data


def get_schema(fs: fsspec.AbstractFileSystem, url: str) -> xr.Dataset:
    logging.info(f"Grabbing schema from {url}")
    with fs.open(url, "rb") as f:
        return vcm.standardize_metadata(xr.open_dataset(f))


if __name__ == '__main__':
    # TODO Parameterize these settings
    url = "gs://vcm-ml-data/2020-03-16-5-day-X-SHiELD-simulation-C384-restart-files/"
    times = list_timesteps(url)

    fs = fsspec.filesystem('gs')
    categories = CATEGORIES
    tiles = [1, 2, 3, 4, 5, 6]

    # get schema for first timestep
    time = times[0]
    schema = xr.merge([
        get_schema(fs, _file(url, time, category, tile=1))
        for category in categories
    ])
    print("Schema:")
    print(schema)

    #
    store = "big.zarr"
    group = zarr.open_group(store, mode="w")
    output_m = fv3net.ZarrMapping(group, schema, dims=['time', 'tile'], coords={'tile': tiles, 'time': times})
    load_timestep = get_timestep(fs, url)
    map_fn = curry(insert_timestep, output_m, load_timestep)

    client = Client()
    results = client.map(map_fn, list(product(times, categories, tiles)))
    # wait for all results to complete
    client.gather(results)
