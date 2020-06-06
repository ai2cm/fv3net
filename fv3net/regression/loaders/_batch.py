import functools
import logging
from itertools import chain
from toolz import partition
from numpy.random import RandomState
import random
from typing import Iterable, Sequence, Mapping, Any, Hashable, TypeVar
import xarray as xr
from vcm import safe
from ._sequences import FunctionOutputSequence
from ._transform import stack_dropnan_shuffle
from .constants import TIME_NAME
from fv3net.regression import loaders

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def batches_from_mapper(
    data_path: str,
    variable_names: Iterable[str],
    mapping_function: str,
    mapping_kwargs: Mapping[str, Any] = None,
    timesteps_per_batch: int = 1,
    num_batches: int = None,
    random_seed: int = 0,
    init_time_dim_name: str = "initial_time",
    rename_variables: Mapping[str, str] = None,
) -> Sequence[xr.Dataset]:
    """ The function returns a sequence of datasets that is later
    iterated over in  ..sklearn.train.

    Args:
        data_mapping (Mapping[str, xr.Dataset]): Interface to select data for
            given timestep keys.
        variable_names (Iterable[str]): data variables to select
        timesteps_per_batch (int, optional): Defaults to 1.
        num_batches (int, optional): Defaults to None.
        random_seed (int, optional): Defaults to 0.
        init_time_dim_name (str, optional): Name of time dim in data source.
            Defaults to "initial_time".
        rename_variables (Mapping[str, str], optional): Defaults to None.

    Raises:
        TypeError: If no variable_names are provided to select the final datasets

    Returns:
        Sequence of xarray datasets for use in training batches.
    """
    num_timesteps = num_batches * timesteps_per_batch
    # TODO maybe raise value error here

    random_state = RandomState(random_seed)
    random.seed(random_seed)
    data_mapping = _create_mapper(data_path, mapping_function, mapping_kwargs)

    times = random.sample(list(data_mapping), num_timesteps)

    subset = Subset(data_mapping, times)
    batched_times = partition(times, timesteps_per_batch)

    batches = _mapper_to_batches(
        subset,
        variable_names,
        random_state,
        batched_times,
        init_time_dim_name,
        rename_variables,
    )
    return batches


class Subset(Mapping):
    def __init__(self, mapping: Mapping, keys: Iterable[Hashable]):
        # TODO add typehints
        # TODO move to _transforms.py
        self._mapping = mapping
        self._keys = list(keys)

    def __getattr__(self, item: str):
        return getattr(self._mapping, item)

    def __getitem__(self, key):
        if key not in self.keys():
            print(self.keys())
            raise KeyError(f"'{key}' not found")
        return self._mapping[key]

    def __iter__(self, key):
        return iter(self.keys())

    def __len__(self):
        return len(self._list(self.keys()))

    def keys(self):
        return self._keys


def _create_mapper(
    data_path, mapping_func_name: str, mapping_kwargs: Mapping[str, Any]
) -> Mapping[str, xr.Dataset]:
    mapping_func = getattr(loaders, mapping_func_name)
    mapping_kwargs = mapping_kwargs or {}
    return mapping_func(data_path, **mapping_kwargs)


def _mapper_to_batches(
    data_mapping: Mapping[str, xr.Dataset],
    variable_names: Iterable[str],
    random_state: RandomState,
    batched_timesteps,
    init_time_dim_name: str = "initial_time",
    rename_variables: Mapping[str, str] = None,
) -> Sequence[xr.Dataset]:
    """ The function returns a sequence of datasets that is later
    iterated over in  ..sklearn.train.
    Args:
        data_mapping (Mapping[str, xr.Dataset]): Interface to select data for
            given timestep keys.
        variable_names (Iterable[str]): data variables to select
        timesteps_per_batch (int, optional): Defaults to 1.
        num_batches (int, optional): Defaults to None.
        random_seed (int, optional): Defaults to 0.
        init_time_dim_name (str, optional): Name of time dim in data source.
            Defaults to "initial_time".
        rename_variables (Mapping[str, str], optional): Defaults to None.
    Raises:
        TypeError: If no variable_names are provided to select the final datasets
    Returns:
        Sequence of xarray datasets
    """
    if rename_variables is None:
        rename_variables = {}
    if len(variable_names) == 0:
        raise TypeError("At least one value must be given for variable_names")
    transform = functools.partial(
        stack_dropnan_shuffle, init_time_dim_name, random_state
    )
    load_batch = functools.partial(
        _load_batch, data_mapping, variable_names, rename_variables, init_time_dim_name,
    )
    return FunctionOutputSequence(lambda x: transform(load_batch(x)), batched_timesteps)


def _load_batch(
    timestep_mapper: Mapping[str, xr.Dataset],
    data_vars: Iterable[str],
    rename_variables: Mapping[str, str],
    init_time_dim_name: str,
    timestep_list: Iterable[str],
) -> xr.Dataset:
    data = _get_dataset_list(timestep_mapper, timestep_list)
    ds = xr.concat(data, init_time_dim_name)
    # need to use standardized time dimension name
    rename_variables[init_time_dim_name] = rename_variables.get(
        init_time_dim_name, TIME_NAME
    )
    ds = ds.rename(rename_variables)
    ds = safe.get_variables(ds, data_vars)
    return ds


def _get_dataset_list(
    timestep_mapper: Mapping[str, xr.Dataset], times: Iterable[str]
) -> Iterable[xr.Dataset]:
    return_list = []
    for time in times:
        ds = timestep_mapper[time]
        return_list.append(ds)
    return return_list


def _validated_num_batches(
    total_num_input_times, timesteps_per_batch, num_batches=None
):
    """ check that the number of batches (if provided) and the number of
    timesteps per batch are reasonable given the number of zarrs in the input data dir.

    Returns:
        Number of batches to use for training
    """
    if any(arg <= 0 for arg in [total_num_input_times, timesteps_per_batch]):
        raise ValueError(
            f"Total number of input times {total_num_input_times}, "
            f"timesteps per batch {timesteps_per_batch}"
        )
    if num_batches is not None and num_batches <= 0:
        raise ValueError(f"num batches {num_batches} cannot be 0 or negative.")
    if num_batches is None:
        if total_num_input_times >= timesteps_per_batch:
            return total_num_input_times // timesteps_per_batch
        else:
            raise ValueError(
                f"Number of input_times {total_num_input_times} "
                f"must be greater than timesteps_per_batch {timesteps_per_batch}"
            )
    elif num_batches * timesteps_per_batch > total_num_input_times:
        raise ValueError(
            f"Number of input_times {total_num_input_times} "
            f"cannot create {num_batches} batches of size {timesteps_per_batch}."
        )
    else:
        return num_batches
