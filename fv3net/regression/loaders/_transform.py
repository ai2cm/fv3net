import xarray as xr
import numpy as np
from typing import Mapping, Tuple
from toolz import groupby

import vcm
from vcm import safe
from ..constants import SAMPLE_DIM_NAME, Z_DIM_NAME

Time = str
Tile = int
K = Tuple[Time, Tile]


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


def stack_and_format(
    init_time_dim_name: str, mask_to_surface_type: str, random, ds: xr.Dataset,
) -> xr.Dataset:
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
    return shuffled(ds, SAMPLE_DIM_NAME, random)


def shuffled(
    dataset: xr.Dataset, dim: str, random: np.random.RandomState
) -> xr.Dataset:
    """
    Shuffles dataset along a dimension within chunks if chunking is present

    Args:
        dataset: input data to be shuffled
        dim: dimension to shuffle indices along
        random: Initialized random number generator state used for shuffling
    """
    chunks_default = (len(dataset[dim]),)
    chunks = dataset.chunks.get(dim, chunks_default)
    chunk_indices = _get_chunk_indices(chunks)
    shuffled_inds = np.concatenate(
        [random.permutation(indices) for indices in chunk_indices]
    )

    return dataset.isel({dim: shuffled_inds})


def _get_chunk_indices(chunks):
    indices = []

    start = 0
    for chunk in chunks:
        indices.append(list(range(start, start + chunk)))
        start += chunk
    return indices
