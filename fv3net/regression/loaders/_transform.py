import numpy as np
from numpy.random import RandomState
from typing import Mapping, Tuple
import xarray as xr
from toolz import groupby

from vcm import safe
from ..constants import SAMPLE_DIM_NAME, Z_DIM_NAMES

Time = str
Tile = int
K = Tuple[Time, Tile]


def stack_dropnan_shuffle(
    init_time_dim_name: str, random_state: RandomState, ds: xr.Dataset,
) -> xr.Dataset:
    ds = ds.load()
    stack_dims = [dim for dim in ds.dims if dim not in Z_DIM_NAMES]
    ds_stacked = safe.stack_once(
        ds,
        SAMPLE_DIM_NAME,
        stack_dims,
        allowed_broadcast_dims=Z_DIM_NAMES + [init_time_dim_name],
    )
    ds_no_nan = ds_stacked.dropna(SAMPLE_DIM_NAME)
    if len(ds_no_nan[SAMPLE_DIM_NAME]) == 0:
        raise ValueError(
            "No Valid samples detected. Check for errors in the training data."
        )
    ds = ds_no_nan.load()
    return shuffled(ds, SAMPLE_DIM_NAME, random_state)


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


class GroupByTime:
    def __init__(self, tiles: Mapping[K, xr.Dataset]) -> Mapping[K, xr.Dataset]:
        def fn(key):
            time, _ = key
            return time

        self._tiles = tiles
        self._time_lookup = groupby(fn, self._tiles.keys())

    def keys(self):
        return list(self._time_lookup.keys())

    def __len__(self):
        return len(self.keys())

    def __getitem__(self, time: Time) -> xr.Dataset:
        datasets = [self._tiles[key] for key in self._time_lookup[time]]
        tiles = range(len(datasets))
        return xr.concat(datasets, dim="tile").assign_coords(tile=tiles)
    
    
class FineResolutionBudgetTiles:
    def __init__(self, fine_resolution_time_mapping: Mapping[Time, xr.Dataset]):
        self._time_mapping = fine_resolution_time_mapping
    
    def keys(self):
        return self._time_mapping.keys()
    
    def __getitem__(self, time: Time) -> xr.Dataset:
        return self._derived_budget_ds(self._time_mapping[Time])
    
    def _derived_budget_ds(
        self,
        budget_time_ds: xr.Dataset,
        variable_prefixes_mapping: Mapping[str]={
            'air_temperature': 'dQ1',
            'specific_humidity': 'dQ2'
        }
    ) -> xr.Dataset:
        
        derived_budget_ds = xr.Dataset()
        for variable, apparent_source in variable_prefixes.items():
            derived_budget_ds.assign({
                apparent_source: (
                    derived_budget_ds[f"{variable}_physics"] + 
                    derived_budget_ds[f"{variable}_microphysics"] + 
                    derived_budget_ds[f"{variable}_convergence"]
                ).assign_attrs({
                    "name": f"apparent source of {variable}",
                    "units": (
                        derived_budget_ds[f"{variable}_physics"]
                        .attrs.get('units', None)
                    )
                })
            })
        
        return derived_budget_ds
    