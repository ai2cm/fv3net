import numpy as np
from numpy.random import RandomState
from typing import Sequence, Union, Optional
import xarray as xr

from vcm import safe


SAMPLE_DIM_NAME = "sample"
DATASET_DIM_NAME = "dataset"
Z_DIM_NAMES = ["z", "pfull"]


class StackedBatches(Sequence[xr.Dataset]):
    def __init__(self, batches: Sequence[xr.Dataset], random_state: RandomState):
        self._batches = batches
        self._random_state = random_state

    def __getitem__(self, idx: Union[int, slice]):
        if isinstance(idx, int):
            return self._stack_batch(self._batches[idx])
        elif isinstance(idx, slice):
            return [self._stack_batch(ds) for ds in self._batches[idx]]
        else:
            raise TypeError(
                f"Invalid argument type of {type(idx)} passed into "
                "StackedBatches.__getitem__."
            )

    def __len__(self) -> int:
        return len(self._batches)

    def _stack_batch(self, ds_unstacked: xr.Dataset) -> xr.Dataset:
        ds = stack_non_vertical(ds_unstacked).load().dropna(dim=SAMPLE_DIM_NAME)
        ds = _check_empty(ds)
        ds = _preserve_samples_per_batch(ds)
        return _shuffled(self._random_state, ds)


def stack_non_vertical(ds: xr.Dataset,) -> xr.Dataset:
    """
    Stack all dimensions except for the Z dimensions into a sample

    Args:
        ds: dataset with geospatial dimensions
    """
    stack_dims = [dim for dim in ds.dims if dim not in Z_DIM_NAMES]
    if len(set(ds.dims).intersection(Z_DIM_NAMES)) > 1:
        raise ValueError("Data cannot have >1 feature dimension in {Z_DIM_NAMES}.")
    ds_stacked = safe.stack_once(
        ds,
        SAMPLE_DIM_NAME,
        stack_dims,
        allowed_broadcast_dims=Z_DIM_NAMES + ["time", "dataset"],
    )
    return ds_stacked.transpose()


def _check_empty(ds: xr.Dataset) -> xr.Dataset:
    """
    Check for an empty variables along a dimension in a dataset
    """
    if len(ds[SAMPLE_DIM_NAME]) == 0:
        raise ValueError("Check for NaN fields in the training data.")
    return ds


def _preserve_samples_per_batch(ds: xr.Dataset) -> xr.Dataset:
    """
    Preserve the approximate number of samples per batch when multiple dataset
    sources are detected in the batch dataset.  Returns an unadjusted dataset
    when no dataset dimension is found.

    Args:
        ds: dataset with sample dimension and potentially a dataset dimension
    """
    try:
        dataset_coord: Optional[xr.DataArray] = ds.coords[DATASET_DIM_NAME]
    except KeyError:
        dataset_coord = None

    if dataset_coord is not None:
        num_datasets = len(set(dataset_coord.values.tolist()))
        ds = ds.thin({SAMPLE_DIM_NAME: num_datasets})

    return ds


def _shuffled(random: RandomState, dataset: xr.Dataset) -> xr.Dataset:
    """
    Shuffles dataset along a dimension within chunks if chunking is present

    Args:
        dim: dimension to shuffle indices along
        random: Initialized random number generator state used for shuffling
        dataset: input data to be shuffled
    """
    chunks_default = (len(dataset[SAMPLE_DIM_NAME]),)
    chunks = dataset.chunks.get(SAMPLE_DIM_NAME, chunks_default)
    chunk_indices = _get_chunk_indices(chunks)
    shuffled_inds = np.concatenate(
        [random.permutation(indices) for indices in chunk_indices]
    )

    return dataset.isel({SAMPLE_DIM_NAME: shuffled_inds})


def _get_chunk_indices(chunks):
    indices = []

    start = 0
    for chunk in chunks:
        indices.append(list(range(start, start + chunk)))
        start += chunk
    return indices
