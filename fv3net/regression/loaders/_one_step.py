import backoff
import functools
import logging
import os
from typing import Iterable, Sequence, Mapping
import copy

import numpy as np
import xarray as xr

import vcm
from vcm import cloud, safe
from ._sequences import FunctionOutputSequence
from ..constants import TIME_NAME, SAMPLE_DIM_NAME, Z_DIM_NAME

__all__ = ["load_one_step_batches"]

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fh = logging.FileHandler("dataset_handler.log")
fh.setLevel(logging.INFO)
logger.addHandler(fh)


class TimestepMapper:
    def __init__(self, timesteps_dir):
        self._timesteps_dir = timesteps_dir
        self._fs = cloud.get_fs(timesteps_dir)
        self.zarrs = self._fs.glob(os.path.join(timesteps_dir, "*.zarr"))
        if len(self.zarrs) == 0:
            raise ValueError(f"No zarrs found in {timesteps_dir}")

    def __getitem__(self, key: str) -> xr.Dataset:
        zarr_path = os.path.join(self._timesteps_dir, f"{key}.zarr")
        return xr.open_zarr(self._fs.get_mapper(zarr_path))

    def keys(self):
        return [vcm.parse_timestep_str_from_path(zarr) for zarr in self.zarrs]

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(self.keys())


def load_one_step_batches(
    data_path: str,
    *variable_names: Iterable[str],
    files_per_batch: int = 1,
    num_batches: int = None,
    random_seed: int = 1234,
    mask_to_surface_type: str = None,
    init_time_dim_name: str = "initial_time",
    rename_variables: Mapping[str, str] = None,
) -> Sequence:
    """Get a sequence of batches from one-step zarr stores.

    Args:
        data_path: location of directory containing zarr stores
        *variable_names: any number of sequences of variable names. One Sequence will be
            returned for each of the given sequences. The "sample" dimension will be
            identical across each of these sequences.
        files_per_batch: number of zarr stores used to create each batch, defaults to 1
        num_batches (optional): number of batches to create. By default, use all the
            available training data.
        random_seed (optional): seed value for random number generator
        mask_to_surface_type: mask data points to ony include the indicated surface type
        init_time_dim_name: name of the initialization time dimension
        rename_variables: mapping of variables to rename,
            from data names to standard names
    """
    if rename_variables is None:
        rename_variables = {}
    if len(variable_names) == 0:
        raise TypeError("At least one value must be given for variable_names")
    logger.info(f"Reading data from {data_path}.")

    timestep_mapper = TimestepMapper(data_path)
    timesteps = timestep_mapper.keys()
    logger.info(f"Number of .zarrs in GCS train data dir: {len(timestep_mapper)}.")
    random = np.random.RandomState(random_seed)
    random.shuffle(timesteps)
    num_batches = _validated_num_batches(len(timesteps), files_per_batch, num_batches)
    logger.info(f"{num_batches} data batches generated for model training.")
    timesteps_list_sequence = list(
        timesteps[batch_num * files_per_batch : (batch_num + 1) * files_per_batch]
        for batch_num in range(num_batches)
    )
    output_list = []
    for data_vars in variable_names:
        load_batch = functools.partial(
            _load_one_step_batch,
            timestep_mapper,
            data_vars,
            rename_variables,
            init_time_dim_name,
        )
        input_formatted_batch = functools.partial(
            stack_and_format,
            init_time_dim_name,
            mask_to_surface_type,
            copy.deepcopy(random),  # each sequence must be shuffled the same!
        )
        output_list.append(
            FunctionOutputSequence(
                lambda x: input_formatted_batch(load_batch(x)), timesteps_list_sequence
            )
        )
    if len(output_list) > 1:
        return tuple(output_list)
    else:
        return output_list[0]


def _load_one_step_batch(
    timestep_mapper,
    data_vars: Iterable[str],
    rename_variables: Mapping[str, str],
    init_time_dim_name: str,
    timestep_list: Iterable[str],
):
    # TODO refactor this I/O. since this logic below it is currently
    # impossible to test.
    data = _load_datasets(timestep_mapper, timestep_list)
    ds = xr.concat(data, init_time_dim_name)
    # need to use standardized time dimension name
    rename_variables[init_time_dim_name] = rename_variables.get(
        init_time_dim_name, TIME_NAME
    )
    ds = ds.rename(rename_variables)
    ds = safe.get_variables(ds, data_vars)
    return ds.load()


@backoff.on_exception(backoff.expo, (ValueError, RuntimeError), max_tries=3)
def _load_datasets(
    timestep_mapper: Mapping[str, xr.Dataset], times: Iterable[str]
) -> Iterable[xr.Dataset]:
    return_list = []
    for time in times:
        ds = timestep_mapper[time]
        return_list.append(ds)
    return return_list


def _validated_num_batches(total_num_input_files, files_per_batch, num_batches=None):
    """ check that the number of batches (if provided) and the number of
    files per batch are reasonable given the number of zarrs in the input data dir.

    Returns:
        Number of batches to use for training
    """
    if num_batches is None:
        num_train_batches = total_num_input_files // files_per_batch
    elif num_batches * files_per_batch > total_num_input_files:
        raise ValueError(
            f"Number of input_files {total_num_input_files} "
            f"cannot create {num_batches} batches of size {files_per_batch}."
        )
    else:
        num_train_batches = num_batches
    return num_train_batches


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
