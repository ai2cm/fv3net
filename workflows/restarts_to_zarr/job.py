from typing import Sequence, Hashable, Iterable, Tuple
import zarr
import os
import xarray as xr
from distributed import Client
from toolz import curry, pipe
from toolz.curried import valmap
import fsspec
import fv3net
from fv3net.pipelines import list_timesteps
import vcm
from vcm.fv3_restarts import _load_restart_lazily, standardize_metadata
from itertools import product
import logging
import dask

logging.basicConfig(level=logging.INFO)

CATEGORIES=('fv_srf_wnd_coarse.res', 'sfc_data_coarse', 'fv_tracer_coarse.res', 'fv_core_coarse.res')


def _files(prefix, category, tiles=(1,2,3,4,5,6)):
    for tile in tiles:
        yield category, tile, prefix + f"{category}.tile{tile}.nc"


def _get_c384_restart_prefix(url, time):
    return os.path.join(url, time, time + '.')


def _files_exist(fs: fsspec.AbstractFileSystem, files: Sequence[str]):
    for dim, file in files:
        if not fs.exists(file):
            return False
    return True


def _load_arrays(fs, files):
    for category, tile, file in files:
        yield tile, _load_restart_lazily(fs, file, category)


def _standardize_metadata(loaded: Iterable[Tuple[int, xr.Dataset]]):
    for tile, ds in loaded:
        yield tile, standardize_metadata(ds)


def _yield_array_sequence(loaded: Iterable[Tuple[int, xr.Dataset]]):
    for tile, ds in loaded:
        for name in ds:
            yield name, (tile,), ds[name]


def open_restarts_prefix(fs: fsspec.AbstractFileSystem, prefix: str, category: str) -> xr.Dataset:
    logging.info(f"Opening {category} at {prefix}")
    return pipe(
        _files(prefix, category),
        curry(_load_arrays, fs),
        _standardize_metadata,
        _yield_array_sequence,
        curry(vcm.combine_array_sequence, labels=['tile'])
    )


@curry
def get_timestep(fs, url, time, category):
    prefix = _get_c384_restart_prefix(url, time)
    return open_restarts_prefix(fs, prefix, category)


def insert_timestep(output: fv3net.ZarrMapping, get, key):
    time, category = key
    with dask.config.set(scheduler='single-threaded'):
        data = get(time, category).load()
    logging.info(f"Setting data at {time} for {category}")
    output[time] = data


if __name__ == '__main__':
    url = "gs://vcm-ml-data/2020-03-16-5-day-X-SHiELD-simulation-C384-restart-files/"
    fs = fsspec.filesystem('gs')
    times = list_timesteps(url)

    # get schema for first timestep
    time = times[0]
    prefix = _get_c384_restart_prefix(url, time)
    CATEGORIES = CATEGORIES[:3]
    times = times[:3]
    schema = xr.merge([
        open_restarts_prefix(fs, prefix, category) for category in CATEGORIES
    ])
    print("Schema:")
    print(schema)

    #
    store = "big.zarr"
    group = zarr.open_group(store, mode="w")
    output_m = fv3net.ZarrMapping(group, schema, times, dim='time')
    load_timestep = get_timestep(fs, url)
    map_fn = curry(insert_timestep, output_m, load_timestep)

    client = Client()
    results = client.map(map_fn, list(product(times, CATEGORIES)))
    # wait for all results to complete
    client.gather(results)


