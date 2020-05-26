import numpy as np
import xarray as xr
import vcm
from vcm import safe
from ..constants import SAMPLE_DIM_NAME, Z_DIM_NAME


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
