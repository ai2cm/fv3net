import functools
import logging
from numpy.random import RandomState
import pandas as pd
from typing import Iterable, Sequence, Mapping, Any, Hashable, Optional, Union, List
import xarray as xr
from vcm import safe, parse_datetime_from_str
from toolz import partition_all, compose
from ._sequences import Map, Local
from .._utils import stack_dropnan_shuffle, get_derived_dataset, nonderived_variables
from ..constants import TIME_NAME
from ._serialized_phys import (
    SerializedSequence,
    FlattenDims,
    open_serialized_physics_data,
)
import loaders

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def batches_from_local(
    data_path: str,
    variable_names: Iterable[str],
    keep_in_memory: bool = False,
    timesteps=None,
) -> Sequence[xr.Dataset]:
    # this wrapper function is needed to transform from a call signature
    # with variable_names to one without.
    return Local(data_path, keep_in_memory=keep_in_memory)


def batches_from_geodata(
    data_path: Union[str, List, tuple],
    variable_names: Iterable[str],
    mapping_function: str,
    mapping_kwargs: Optional[Mapping[str, Any]] = None,
    timesteps_per_batch: int = 1,
    random_seed: int = 0,
    timesteps: Optional[Sequence[str]] = None,
    res: str = "c48",
) -> Sequence[xr.Dataset]:
    """ The function returns a sequence of datasets that is later
    iterated over in  ..sklearn.train. The data is assumed to
    have geospatial dimensions and is accessed through a mapper interface.


    Args:
        data_path (str): Path to data store to be loaded via mapper.
        variable_names (Iterable[str]): data variables to select
        mapping_function (str): Name of a callable which opens a mapper to the data
        mapping_kwargs (Mapping[str, Any]): mapping of keyword arguments to be
            passed to the mapping function
        timesteps_per_batch (int, optional): Defaults to 1.
        random_seed (int, optional): Defaults to 0.
        res: grid resolution, format as f'c{number cells in tile}'
    Raises:
        TypeError: If no variable_names are provided to select the final datasets

    Returns:
        Sequence of xarray datasets for use in training batches.
    """
    data_mapping = _create_mapper(data_path, mapping_function, mapping_kwargs)
    batches = batches_from_mapper(
        data_mapping, variable_names, timesteps_per_batch, random_seed, timesteps, res,
    )
    return batches


def _create_mapper(
    data_path, mapping_func_name: str, mapping_kwargs: Mapping[str, Any]
) -> Mapping[str, xr.Dataset]:
    mapping_func = getattr(loaders.mappers, mapping_func_name)
    mapping_kwargs = mapping_kwargs or {}
    return mapping_func(data_path, **mapping_kwargs)


def batches_from_mapper(
    data_mapping: Mapping[str, xr.Dataset],
    variable_names: Iterable[str],
    timesteps_per_batch: int = 1,
    random_seed: int = 0,
    timesteps: Optional[Sequence[str]] = None,
    res: str = "c48",
) -> Sequence[xr.Dataset]:
    """ The function returns a sequence of datasets that is later
    iterated over in  ..sklearn.train.

    Args:
        data_mapping (Mapping[str, xr.Dataset]): Interface to select data for
            given timestep keys.
        variable_names (Iterable[str]): data variables to select
        timesteps_per_batch (int, optional): Defaults to 1.
        random_seed (int, optional): Defaults to 0.
        timesteps: List of timesteps to use in training.
        res: grid resolution, format as f'c{number cells in tile}'
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
    if len(variable_names) == 0:
        raise TypeError("At least one value must be given for variable_names")

    timesteps = timesteps or data_mapping.keys()
    num_times = len(timesteps)
    times = _sample(timesteps, num_times, random_state)
    batched_timesteps = list(partition_all(timesteps_per_batch, times))

    load_batch = functools.partial(_load_batch, data_mapping, variable_names,)

    transform = functools.partial(stack_dropnan_shuffle, random_state)
    load_batch = functools.partial(_load_batch, data_mapping, variable_names)
    derived_dataset = functools.partial(get_derived_dataset, variable_names, res)
    batch_func = compose(transform, derived_dataset, load_batch)

    seq = Map(batch_func, batched_timesteps)
    seq.attrs["times"] = times

    return seq


def diagnostic_batches_from_geodata(
    data_path: Union[str, List, tuple],
    variable_names: Sequence[str],
    mapping_function: str,
    mapping_kwargs: Optional[Mapping[str, Any]] = None,
    timesteps_per_batch: int = 1,
    random_seed: int = 0,
    timesteps: Optional[Sequence[str]] = None,
    res: str = "c48",
) -> Sequence[xr.Dataset]:
    """Load a dataset sequence for dagnostic purposes. Uses the same batch subsetting as
    as batches_from_mapper but without transformation and stacking

    Args:
        data_path: Path to data store to be loaded via mapper.
        variable_names (Iterable[str]): data variables to select
        mapping_function (str): Name of a callable which opens a mapper to the data
        mapping_kwargs (Mapping[str, Any]): mapping of keyword arguments to be
            passed to the mapping function
        timesteps_per_batch (int, optional): Defaults to 1.
        num_batches (int, optional): Defaults to None.
        random_seed (int, optional): Defaults to 0.
        timesteps: List of timesteps to use in training.
        res: grid resolution, format as f'c{number cells in tile}'

    Raises:
        TypeError: If no variable_names are provided to select the final datasets

    Returns:
        Sequence of xarray datasets for use in training batches.
    """

    data_mapping = _create_mapper(data_path, mapping_function, mapping_kwargs)
    sequence = diagnostic_batches_from_mapper(
        data_mapping, variable_names, timesteps_per_batch, random_seed, timesteps, res,
    )
    return sequence


def diagnostic_batches_from_mapper(
    data_mapping: Mapping[str, xr.Dataset],
    variable_names: Sequence[str],
    timesteps_per_batch: int = 1,
    random_seed: int = 0,
    timesteps: Sequence[str] = None,
    res: str = "c48",
) -> Sequence[xr.Dataset]:
    if timesteps and set(timesteps).issubset(data_mapping.keys()) is False:
        raise ValueError(
            "Timesteps specified in file are not present in data: "
            f"{list(set(timesteps)-set(data_mapping.keys()))}"
        )
    random_state = RandomState(random_seed)
    timesteps = timesteps or data_mapping.keys()
    num_times = len(timesteps)
    times = _sample(timesteps, num_times, random_state)
    batched_timesteps = list(partition_all(timesteps_per_batch, times))

    load_batch = functools.partial(_load_batch, data_mapping, variable_names)
    derived_dataset = functools.partial(get_derived_dataset, variable_names, res,)
    batch_func = compose(derived_dataset, load_batch)
    seq = Map(batch_func, batched_timesteps)
    seq.attrs["times"] = times
    return seq


def _sample(seq: Sequence[Any], n: int, random_state: RandomState) -> Sequence[Any]:
    return random_state.choice(list(seq), n, replace=False).tolist()


def _load_batch(
    mapper: Mapping[str, xr.Dataset],
    data_vars: Iterable[str],
    keys: Iterable[Hashable],
) -> xr.Dataset:

    time_coords = [parse_datetime_from_str(key) for key in keys]
    ds = xr.concat([mapper[key] for key in keys], pd.Index(time_coords, name=TIME_NAME))
    nonderived_vars = nonderived_variables(data_vars, ds.data_vars)
    ds = safe.get_variables(ds, nonderived_vars)
    return ds


def batches_from_serialized(
    path: str,
    zarr_prefix: str = "phys",
    sample_dims: Sequence[str] = ["savepoint", "rank", "horizontal_dimension"],
    savepoints_per_batch: int = 1,
) -> Map:
    """
    Load a sequence of serialized physics data for use in model fitting procedures.
    Data variables are reduced to a sample and feature dimension by stacking specified
    dimensions any remaining feature dims along the last dimension. (An extra last
    dimensiononly appeared for tracer fields in the serialized turbulence data.)

    Args:
        path: Path (local or remote) to the input/output zarr files
        zarr_prefix: Zarr file prefix for input/output files.  Becomes {prefix}_in.
            zarr and {prefix}_out.zarr
        sample_dims: Sequence of dimensions to stack as a single sample dimension
        savepoints_per_batch: Number of serialized savepoints to include in a single
            batch
    
    Returns:
        A seqence of batched serialized data ready for model testing/training
    """
    ds = open_serialized_physics_data(path, zarr_prefix=zarr_prefix)
    seq = SerializedSequence(ds)
    seq = FlattenDims(seq, sample_dims)

    if savepoints_per_batch > 1:
        batch_args = [
            slice(start, start + savepoints_per_batch)
            for start in range(0, len(seq), savepoints_per_batch)
        ]
    else:
        batch_args = list(range(len(seq)))

    def _load_item(item: Union[int, slice]):
        return seq[item]

    func_seq = Map(_load_item, batch_args)

    return func_seq
