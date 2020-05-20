import os
import zarr
import xarray as xr
import numpy as np
from typing import Sequence, Iterable, Mapping
from functools import partial

import vcm
from vcm import cloud, safe
from ._sequences import FunctionOutputSequence

INPUT_ZARR = "before_dynamics.zarr"
NUDGING_TENDENCY_ZARR = "nudging_tendencies.zarr"
TIMESCALE_OUTDIR_TEMPLATE = "outdir-{}h"

SAMPLE_DIM = "sample"


def load_nudging_batches(
    data_path: str,
    input_variables: Iterable[str],
    output_variables: Iterable[str],
    nudging_timescale: int,
    num_samples_in_batch: int = 13824,
    num_batches: int = None,
    random_seed: int = 0,
    mask_to_surface_type: str = None,
    z_dim_name: str = "z",
    rename_variables: Mapping[str, str] = None,
    time_dim_name: str = "time",
    initial_time_skip: int = 0,
    include_ntimes: int = None,
) -> Sequence:
    """
    Get a sequence of batches from a nudged-run zarr store.

    Args:
        data_path: location of directory containing zarr stores
        input_variables: names of inputs
        output_variables: names of outputs
        nudging_timescale: timescale of the nudging for the simulation
            being used as input.
        num_samples_in_batch (optional): number of samples to include
            in a single batch item.  Overridden by num_batches
        num_batches (optional): number of batches to split the
            input samples into.  Overrides num_samples_in_batch.
        random_seed (optional): A seed for the RNG state used in shuffling operations
        mask_to_surface_type (optional): Flag selector for surface type masking.
            Requires "land_sea_mask" exists in the loaded dataset.  Note: as currently
            implemented NaN drop may reduce the batch size under requested
            number of samples.
        z_dim_name (optional): vertical dimension name to retain in the dimension
            stacking procedure
        rename_variables (optional): A mapping to update any variable names in the
            dataset prior to the selection of input/output variables
        time_dim
        initial_time_skip (optional): Number of initial time indices to skip to avoid
            spin-up samples
        include_ntimes (optional): Number of times (by index) to include in the
            batch resampling operation
    """
    data_path = os.path.join(
        data_path, TIMESCALE_OUTDIR_TEMPLATE.format(nudging_timescale),
    )

    combined = _load_nudging_zarr(
        data_path, input_variables, output_variables, rename_variables
    )

    start = initial_time_skip
    end = start + include_ntimes
    combined = combined.isel({time_dim_name: slice(start, end)})

    if mask_to_surface_type is not None:
        combined = vcm.mask_to_surface_type(combined, mask_to_surface_type)

    stack_dims = [dim for dim in combined.dims if dim != z_dim_name]
    combined_stacked = safe.stack_once(
        combined, SAMPLE_DIM, stack_dims, allowed_broadcast_dims=[z_dim_name]
    )

    total_samples = combined_stacked.sizes[SAMPLE_DIM]

    func_args = _get_batch_func_args(
        total_samples, num_samples_in_batch, num_batches=num_batches
    )
    random = np.random.RandomState(random_seed)
    random.shuffle(func_args)

    loader_func = partial(_load_nudging_batch, combined_stacked, random)

    return FunctionOutputSequence(loader_func, func_args)


def _load_nudging_batch(
    stacked_ds: xr.Dataset, random: np.random.RandomState, batch_slice: slice
) -> xr.Dataset:

    batch = stacked_ds.isel({SAMPLE_DIM: batch_slice})
    batch_no_nan = batch.dropna(SAMPLE_DIM)

    if len(batch_no_nan[SAMPLE_DIM]) == 0:
        raise ValueError(
            "No Valid samples detected. Check for errors in the training data."
        )

    # TODO: backoff here?
    final_batch = batch_no_nan.load()  # TODO: Or is it already loaded from dropna?
    return _shuffled(final_batch, SAMPLE_DIM, random)


def _get_batch_func_args(
    num_samples: int, samples_per_batch: int, num_batches: int = None
):

    if num_batches is not None:
        batch_size = num_samples // num_batches
    else:
        batch_size = samples_per_batch
        num_batches = num_samples // samples_per_batch

    if batch_size == 0 or num_batches == 0:
        raise ValueError(
            "The number of input samples is insufficient to create a batch for "
            f"requested samples_per_batch={samples_per_batch} or "
            f"requested num_batches={num_batches}."
        )

    slices = []
    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size
        slices.append(slice(start, end))

    args = [(item,) for item in slices]

    return args


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


def _load_nudging_zarr(path, input_vars, output_vars, rename_variables):
    fs = cloud.get_fs(path)
    input_data = _open_zarr(fs, os.path.join(path, INPUT_ZARR))
    input_data = _rename_ds_variables(input_data, rename_variables)
    output_data = _open_zarr(fs, os.path.join(path, NUDGING_TENDENCY_ZARR))
    output_data = _rename_ds_variables(output_data, rename_variables)

    inputs = safe.get_variables(input_data, input_vars)
    outputs = safe.get_variables(output_data, output_vars)

    combined = xr.merge([inputs, outputs], join="inner")

    return combined


def _open_zarr(fs, url):

    mapper = fs.get_mapper(url)
    cached_mapper = zarr.storage.LRUStoreCache(mapper, max_size=None)

    return xr.open_zarr(cached_mapper)


def _rename_ds_variables(ds, rename_variables):

    to_rename = {
        var_name: darray
        for var_name, darray in rename_variables.items()
        if var_name in ds
    }
    ds = ds.rename(to_rename)

    return ds
