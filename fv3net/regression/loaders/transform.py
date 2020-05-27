from functools import partial
import numpy as np
from typing import Callable, Mapping, Tuple, List
import xarray as xr
from toolz import groupby

import vcm
from vcm import safe
from ..constants import TIME_NAME, SAMPLE_DIM_NAME, Z_DIM_NAME

Time = str
Tile = int
K = Tuple[Time, Tile]


def construct_data_transform(transform_configs) -> Callable:
    transform = lambda x: x
    for transform_config in transform_configs:
        transform_name, args = list(transform_config.items())[0]
        transform_func = globals()[transform_name]
        partial_transform = partial(transform_func, *args)
        transform = _compose(partial_transform, transform)
    return transform
           

def _compose(outer_func, inner_func):
    return lambda x: outer_func(inner_func(x))


# TODO: reformat to clean separation of the three transforms that
# can be passed in order to the TransformData object
def mask_stack_shuffle(
    init_time_dim_name: str, mask_to_surface_type: str, random_seed: int, ds: xr.Dataset,
) -> xr.Dataset:
    random = np.random.RandomState(random_seed)
    if mask_to_surface_type is not None:
        ds = vcm.mask_to_surface_type(ds, mask_to_surface_type)
    stack_dims = [dim for dim in ds.dims if dim != Z_DIM_NAME]
    ds_stacked = safe.stack_once(
        ds,
        SAMPLE_DIM_NAME,
        stack_dims,
        allowed_broadcast_dims=[Z_DIM_NAME, init_time_dim_name],
    )

    ds_no_nan = ds_stacked.dropna(SAMPLE_DIM_NAME)

    if len(ds_no_nan[SAMPLE_DIM_NAME]) == 0:
        raise ValueError(
            "No Valid samples detected. Check for errors in the training data."
        )
    ds = ds_no_nan.load()
    return _shuffled(ds, SAMPLE_DIM_NAME, random)


def _shuffled(dataset, dim, random):
    chunks_default = (len(dataset[dim]),)
    chunks = dataset.chunks.get(dim, chunks_default)
    indices = _chunk_indices(chunks)
    shuffled_inds = _shuffled_within_chunks(indices, random)
    return dataset.isel({dim: shuffled_inds})


def _chunk_indices(chunks):
    indices = []

    start = 0
    for chunk in chunks:
        indices.append(list(range(start, start + chunk)))
        start += chunk
    return indices


def _shuffled_within_chunks(indices, random):
    # We should only need to set the random seed once (not every time)
    return np.concatenate([random.permutation(index) for index in indices])


class GroupByTime:
    def __init__(self, tiles: Mapping[K, xr.Dataset]) -> Mapping[K, xr.Dataset]:
        def fn(key):
            time, _ = key
            return time

        self._tiles = tiles
        self._time_lookup = groupby(fn, self._tiles.keys())

    def keys(self):
        return self._time_lookup.keys()

    def __len__(self):
        return len(self.keys())

    def __getitem__(self, time: Time) -> xr.Dataset:
        datasets = [self._tiles[key] for key in self._time_lookup[time]]
        tiles = range(len(datasets))
        return xr.concat(datasets, dim="tile").assign_coords(tile=tiles)
