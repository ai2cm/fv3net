import functools
import logging
from numpy.random import RandomState
from typing import Iterable, Sequence, Mapping, Any, Hashable
import xarray as xr
from vcm import safe
from toolz import partition
from ._sequences import FunctionOutputSequence
from ._transform import stack_dropnan_shuffle
from .constants import TIME_NAME
from fv3net.regression import loaders


logger = logging.getLogger(__name__)
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
    timesteps: Sequence[str] = None,
) -> Sequence[xr.Dataset]:
    """ The function returns a sequence of datasets that is later
    iterated over in  ..sklearn.train.

    Args:
        data_path (str): Path to data store to be loaded via mapper.
        variable_names (Iterable[str]): data variables to select
        mapping_function (str): Name of a callable which opens a mapper to the data
        mapping_kwargs (Mapping[str, Any]): mapping of keyword arguments to be
            passed to the mapping function
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
    data_mapping = _create_mapper(data_path, mapping_function, mapping_kwargs)
    batches = _mapper_to_batches(
        data_mapping,
        variable_names,
        timesteps_per_batch,
        num_batches,
        random_seed,
        init_time_dim_name,
        rename_variables,
        timesteps,
    )
    return batches


def _create_mapper(
    data_path, mapping_func_name: str, mapping_kwargs: Mapping[str, Any]
) -> Mapping[str, xr.Dataset]:
    mapping_func = getattr(loaders, mapping_func_name)
    mapping_kwargs = mapping_kwargs or {}
    return mapping_func(data_path, **mapping_kwargs)


def _mapper_to_batches(
    data_mapping: Mapping[str, xr.Dataset],
    variable_names: Iterable[str],
    timesteps_per_batch: int = 1,
    num_batches: int = None,
    random_seed: int = 0,
    init_time_dim_name: str = "initial_time",
    rename_variables: Mapping[str, str] = None,
    timesteps: Sequence[str] = None,
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
        timesteps (Sequence[str], optional): List of timesteps to use in training.
    Raises:
        TypeError: If no variable_names are provided to select the final datasets
    Returns:
        Sequence of xarray datasets
    """
    if timesteps and set(timesteps).issubset(data_mapping.keys()) is False:
        raise ValueError(
            "Timesteps specified in file are not present in data: "
            f"{list(set(timesteps)-set(data_mapping.keys()))}"
        )

    random_state = RandomState(random_seed)
    if rename_variables is None:
        rename_variables = {}
    if len(variable_names) == 0:
        raise TypeError("At least one value must be given for variable_names")

    num_batches = _validated_num_batches(
        len(timesteps or data_mapping), timesteps_per_batch, num_batches
    )
    num_times = timesteps_per_batch * num_batches
    times = _sample(timesteps or data_mapping.keys(), num_times, random_state)
    batched_timesteps = list(partition(timesteps_per_batch, times))

    transform = functools.partial(
        stack_dropnan_shuffle, init_time_dim_name, random_state
    )
    load_batch = functools.partial(
        _load_batch, data_mapping, variable_names, rename_variables, init_time_dim_name,
    )
    seq = FunctionOutputSequence(lambda x: transform(load_batch(x)), batched_timesteps)
    seq.attrs["times"] = times

    return seq


def diagnostic_sequence_from_mapper(
    data_path: str,
    variable_names: Sequence[str],
    mapping_function: str,
    mapping_kwargs: Mapping[str, Any] = None,
    timesteps_per_batch: int = 1,
    num_batches: int = None,
    random_seed: int = 0,
    init_time_dim_name: str = "initial_time",
    rename_variables: Mapping[str, str] = None,
) -> Sequence[xr.Dataset]:
    """Load a dataset sequence for dagnostic purposes. Uses the same batch subsetting as
    as batches_from_mapper but without transformation and stacking
    Args:
        data_path (str): Path to data store to be loaded via mapper.
        variable_names (Iterable[str]): data variables to select
        mapping_function (str): Name of a callable which opens a mapper to the data
        mapping_kwargs (Mapping[str, Any]): mapping of keyword arguments to be
            passed to the mapping function
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

    data_mapping = _create_mapper(data_path, mapping_function, mapping_kwargs)

    sequence = _mapper_to_diagnostic_sequence(
        data_mapping,
        variable_names,
        timesteps_per_batch,
        num_batches,
        random_seed,
        init_time_dim_name,
        rename_variables,
    )

    return sequence


def _mapper_to_diagnostic_sequence(
    data_mapping: Mapping[str, xr.Dataset],
    variable_names: Sequence[str],
    timesteps_per_batch: int = 1,
    num_batches: int = None,
    random_seed: int = 0,
    init_time_dim_name: str = "initial_time",
    rename_variables: Mapping[str, str] = None,
    timesteps: Sequence[str] = None,
) -> Sequence[xr.Dataset]:
    if timesteps and set(timesteps).issubset(data_mapping.keys()) is False:
        raise ValueError(
            "Timesteps specified in file are not present in data: "
            f"{list(set(timesteps)-set(data_mapping.keys()))}"
        )
    random_state = RandomState(random_seed)
    if rename_variables is None:
        rename_variables = {}

    num_batches = _validated_num_batches(
        len(timesteps or data_mapping), timesteps_per_batch, num_batches
    )
    num_times = timesteps_per_batch * num_batches
    times = _sample(data_mapping.keys(), num_times, random_state)
    batched_timesteps = list(partition(timesteps_per_batch, times))

    load_batch = functools.partial(
        _load_batch, data_mapping, variable_names, rename_variables, init_time_dim_name,
    )
    seq = FunctionOutputSequence(load_batch, batched_timesteps)
    seq.attrs["times"] = times
    return seq


def _sample(seq: Sequence[Any], n: int, random_state: RandomState) -> Sequence[Any]:
    return random_state.choice(list(seq), n, replace=False).tolist()


def _load_batch(
    mapper: Mapping[str, xr.Dataset],
    data_vars: Iterable[str],
    rename_variables: Mapping[str, str],
    init_time_dim_name: str,
    keys: Iterable[Hashable],
) -> xr.Dataset:
    ds = xr.concat([mapper[key] for key in keys], init_time_dim_name)
    # need to use standardized time dimension name
    rename_variables[init_time_dim_name] = rename_variables.get(
        init_time_dim_name, TIME_NAME
    )
    ds = ds.rename(rename_variables)
    ds = safe.get_variables(ds, data_vars)
    return ds


def _validated_num_batches(
    total_num_input_times, timesteps_per_batch, num_batches=None
):
    """ Checks that the number of batches (if provided) and the number of
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
