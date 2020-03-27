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


def _files(prefix, categories=('fv_srf_wnd_coarse.res', 'sfc_data_coarse', 'fv_tracer_coarse.res', 'fv_core_coarse.res'), tiles=(1,2,3,4,5,6)):
    for category, tile in product(categories, tiles):
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


def open_restarts_prefix(fs: fsspec.AbstractFileSystem, prefix: str, **kwargs) -> xr.Dataset:
    return pipe(
        _files(prefix, **kwargs),
        curry(_load_arrays, fs),
        _standardize_metadata,
        _yield_array_sequence,
        curry(vcm.combine_array_sequence, labels=['tile'])
    )


@curry
def get_timestep(fs, url, timesteps, index):
    return open_restarts_prefix(fs, url, timesteps[index])


def insert_timestep(output: fv3net.ZarrMapping, get, time: Hashable):
    output[time] = get(time)


if __name__ == '__main__':
    url = "gs://vcm-ml-data/2020-03-16-5-day-X-SHiELD-simulation-C384-restart-files/"
    fs = fsspec.filesystem('gs')
    times = list_timesteps(url)

    # get schema for first timestep
    time = times[0]
    prefix = _get_c384_restart_prefix(url, time)
    schema = open_restarts_prefix(fs, prefix)

    #
    times = times[:2]
    store = "big.zarr"
    group = zarr.open_group(store)
    output_m = fv3net.ZarrMapping(group, schema, times, dim='time')
    load_timestep = get_timestep(fs, url, times)
    map_fn = curry(insert_timestep, output_m, load_timestep)

    client = Client()
    for time in enumerate(times):
        client.submit(map_fn, time)


