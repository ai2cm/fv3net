import functools
import logging
from itertools import chain
from toolz import partition, compose
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


def random_sample(mapper, random_seed, num_batches, batch_size):
    # TODO move to _transform.py
    # TODO type hint
    num_timesteps = num_batches * batch_size
    # TODO maybe raise value error here
    random.seed(random_seed)
    times = random.sample(list(mapper), num_timesteps)
    return Subset(mapper, times)


def batches_from_mapper(
    # TODO add type hint
    data_mapping,
    variable_names: Iterable[str],
    timesteps_per_batch: int = 1,
    num_batches: int = None,
    random_seed: int = 0,
    init_time_dim_name: str = "initial_time",
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

    Raises:
        TypeError: If no variable_names are provided to select the final datasets

    Returns:
        Sequence of xarray datasets for use in training batches.
    """
    random_state = RandomState(random_seed)
    # These code depends only on subset
    batched_times = list(partition(timesteps_per_batch, list(data_mapping.keys())))
    transform = functools.partial(
        stack_dropnan_shuffle, init_time_dim_name, random_state
    )
    load_batch = functools.partial(
        _load_batch, data_mapping, variable_names, init_time_dim_name,
    )
    return FunctionOutputSequence(lambda x: transform(load_batch(x)), batched_times)


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


def _load_batch(
    timestep_mapper: Mapping[str, xr.Dataset],
    data_vars: Iterable[str],
    init_time_dim_name: str,
    timestep_list: Iterable[str],
) -> xr.Dataset:
    data = _get_dataset_list(timestep_mapper, timestep_list)
    ds = xr.concat(data, init_time_dim_name)
    # need to use standardized time dimension name
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
