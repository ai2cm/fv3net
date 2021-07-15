import numpy as np
from numpy.random import RandomState
from typing import List, Sequence, Union, Optional
import xarray as xr

from vcm import safe


SAMPLE_DIM_NAME = "sample"
DATASET_DIM_NAME = "dataset"
Z_DIM_NAMES = ["z", "pfull"]


def parse_data_path(data_path: Union[List, str]):
    # allows the data path to be provided as a sequence of urls,
    # which is useful for hybrid training data
    if isinstance(data_path, List) and len(data_path) == 1:
        return data_path[0]
    else:
        return data_path


def stack_batches(
    batches: Sequence[xr.Dataset], random_state: RandomState
) -> Sequence[xr.Dataset]:
    for ds in batches:
        yield stack_dataset(ds, random_state)


def stack_dataset(ds_unstacked: xr.Dataset, random_state: RandomState) -> xr.Dataset:
    ds = _stack_non_vertical(ds_unstacked).load().dropna(dim=SAMPLE_DIM_NAME)
    ds = _check_empty(ds)
    ds = _preserve_samples_per_batch(ds)
    return _shuffled(ds, random_state)


def _stack_non_vertical(ds: xr.Dataset,) -> xr.Dataset:
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
