import backoff
import functools
import logging
import numpy as np
from typing import Iterable, Sequence, Mapping, Callable
import xarray as xr
from vcm import safe
from ._sequences import FunctionOutputSequence
from ..constants import TIME_NAME, SAMPLE_DIM_NAME, Z_DIM_NAME

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def load_batches(
        data_mapping: Mapping[str, xr.Dataset],
        transform_func: Callable,
        *variable_names: Iterable[str],
        files_per_batch: int = 1,
        num_batches: int = None,
        random_seed: int = 0,
        init_time_dim_name: str = "initial_time",
        rename_variables: Mapping[str, str] = None,
):
    batched_timesteps = _select_batch_timesteps(
        data_mapping.keys(),
        files_per_batch,
        num_batches,
        random_seed)
    output_list = []
    for data_vars in variable_names:
        partial_load_batch = functools.partial(
            _load_batch,
            data_mapping,
            data_vars,
            rename_variables,
            init_time_dim_name,
        )
        output_list.append(
            FunctionOutputSequence(
                lambda x: transform_func(partial_load_batch(x)), batched_timesteps
            )
        )
    return output_list


def _select_batch_timesteps(
        timesteps: Sequence[str],
        files_per_batch,
        num_batches,
        random_seed,
) -> Sequence[Sequence[str]]:
    random = np.random.RandomState(random_seed)
    random.shuffle(timesteps)
    num_batches = _validated_num_batches(len(timesteps), files_per_batch, num_batches)
    logger.info(f"{num_batches} data batches generated for model training.")
    timesteps_list_sequence = list(
        timesteps[batch_num * files_per_batch : (batch_num + 1) * files_per_batch]
        for batch_num in range(num_batches)
    ) 
    return timesteps_list_sequence


def _load_batch(
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
