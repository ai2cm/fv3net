import logging
from loaders.typing import Batches
import numpy as np
import pandas as pd
from typing import (
    Iterable,
    Sequence,
    Mapping,
    Any,
    Optional,
    Union,
    List,
)
import xarray as xr
from vcm import safe, parse_datetime_from_str
from toolz import partition_all, curry, compose_left
from ._sequences import Map
from .._utils import (
    add_grid_info,
    add_derived_data,
    add_wind_rotation_info,
    nonderived_variables,
    stack,
    # shuffle,
    dropna,
    # select_first_samples,
    select_random_ordered_subset,
)
from ..constants import TIME_NAME
from .._config import batches_functions, batches_from_mapper_functions
from ._serialized_phys import (
    SerializedSequence,
    FlattenDims,
    open_serialized_physics_data,
)
import loaders
import fsspec
import vcm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@batches_functions.register
def batches_from_geodata(
    data_path: Union[str, List, tuple],
    variable_names: Iterable[str],
    mapping_function: str,
    mapping_kwargs: Optional[Mapping[str, Any]] = None,
    timesteps_per_batch: int = 1,
    timesteps: Optional[Sequence[str]] = None,
    res: str = "c48",
    needs_grid: bool = True,
    in_memory: bool = False,
    unstacked_dims: Optional[Sequence[str]] = None,
    subsample_ratio: float = 1.0,
    drop_nans: bool = False,
) -> loaders.typing.Batches:
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
        needs_grid: Add grid information into batched datasets. [Warning] requires
            remote GCS access
        in_memory: if True, load data eagerly and keep it in memory
        unstacked_dims: if given, produce stacked and shuffled batches retaining
            these dimensions as unstacked (non-sample) dimensions
        subsample_ratio: the fraction of data to retain in each batch, selected
            at random along the sample dimension.
        drop_nans: if True, drop samples with NaN values from the data, and raise an
            exception if all values in a batch are NaN. requires unstacked_dims
            argument is given, raises a ValueError otherwise.
    Raises:
        TypeError: If no variable_names are provided to select the final datasets

    Returns:
        Sequence of xarray datasets for use in training batches.
    """
    if mapping_kwargs is None:
        mapping_kwargs = {}
    data_mapping = _create_mapper(data_path, mapping_function, mapping_kwargs)
    batches = batches_from_mapper(
        data_mapping,
        variable_names=variable_names,
        timesteps_per_batch=timesteps_per_batch,
        timesteps=timesteps,
        res=res,
        needs_grid=needs_grid,
        in_memory=in_memory,
        unstacked_dims=unstacked_dims,
        subsample_ratio=subsample_ratio,
        drop_nans=drop_nans,
    )
    return batches


def _create_mapper(
    data_path, mapping_func_name: str, mapping_kwargs: Mapping[str, Any]
) -> Mapping[str, xr.Dataset]:
    mapping_func = getattr(loaders.mappers, mapping_func_name)
    return mapping_func(data_path, **mapping_kwargs)


@batches_from_mapper_functions.register
def batches_from_mapper(
    data_mapping: Mapping[str, xr.Dataset],
    variable_names: Sequence[str],
    timesteps_per_batch: int = 1,
    timesteps: Optional[Sequence[str]] = None,
    res: str = "c48",
    needs_grid: bool = True,
    in_memory: bool = False,
    unstacked_dims: Optional[Sequence[str]] = None,
    subsample_ratio: float = 1.0,
    drop_nans: bool = False,
) -> loaders.typing.Batches:
    """ The function returns a sequence of datasets that is later
    iterated over in  ..sklearn.train.

    Args:
        data_mapping: Interface to select data for
            given timestep keys.
        variable_names: data variables to select
        timesteps_per_batch (int, optional): Defaults to 1.
        timesteps: List of timesteps to use in training.
        needs_grid: Add grid information into batched datasets. [Warning] requires
            remote GCS access
        in_memory: if True, load data eagerly and keep it in memory
        unstacked_dims: if given, produce stacked and shuffled batches retaining
            these dimensions as unstacked (non-sample) dimensions
        subsample_ratio: the fraction of data to retain in each batch, selected
            at random along the sample dimension.
        drop_nans: if True, drop samples with NaN values from the data, and raise an
            exception if all values in a batch are NaN. requires unstacked_dims
            argument is given, raises a ValueError otherwise.
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

    if len(variable_names) == 0:
        raise TypeError("At least one value must be given for variable_names")

    if timesteps is None:
        timesteps = list(data_mapping.keys())
    shuffled_times = np.random.choice(timesteps, len(timesteps), replace=False).tolist()
    batched_timesteps = list(partition_all(timesteps_per_batch, shuffled_times))

    # First function goes from mapper + timesteps to xr.dataset
    # Subsequent transforms are all dataset -> dataset
    transforms = [_get_batch(data_mapping, variable_names)]

    if needs_grid:
        transforms += [
            add_grid_info(res),
            add_wind_rotation_info(res),
        ]

    transforms.append(add_derived_data(variable_names))

    if unstacked_dims is not None:
        transforms.append(curry(stack)(unstacked_dims))
        # transforms.append(shuffle)
        # transforms.append(select_first_samples(subsample_ratio))
        transforms.append(select_random_ordered_subset(subsample_ratio))
        if drop_nans:
            transforms.append(dropna)
    elif subsample_ratio != 1.0:
        raise ValueError(
            "setting subsample_ratio != 1.0 requires providing unstacked_dims"
        )
    elif drop_nans:
        raise ValueError("drop_nans=True requires unstacked_dims argument is provided")

    batch_func = compose_left(*transforms)

    seq = Map(batch_func, batched_timesteps)
    seq.attrs["times"] = shuffled_times

    if in_memory:
        out_seq: Batches = tuple(ds.load() for ds in seq)
    else:
        out_seq = seq
    return out_seq


@batches_functions.register
def diagnostic_batches_from_geodata(
    data_path: Union[str, List, tuple],
    variable_names: Sequence[str],
    mapping_function: str,
    mapping_kwargs: Optional[Mapping[str, Any]] = None,
    timesteps_per_batch: int = 1,
    random_seed: int = 0,
    timesteps: Optional[Sequence[str]] = None,
    res: str = "c48",
    subsample_size: int = None,
    needs_grid: bool = True,
) -> loaders.typing.Batches:
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
        needs_grid: Add grid information into batched datasets. [Warning] requires
            remote GCS access

    Raises:
        TypeError: If no variable_names are provided to select the final datasets

    Returns:
        Sequence of xarray datasets for use in training batches.
    """
    if mapping_kwargs is None:
        mapping_kwargs = {}
    data_mapping = _create_mapper(data_path, mapping_function, mapping_kwargs)
    sequence = batches_from_mapper(
        data_mapping,
        variable_names,
        timesteps_per_batch,
        random_seed,
        timesteps,
        res,
        needs_grid=needs_grid,
    )
    return sequence


@curry
def _get_batch(
    mapper: Mapping[str, xr.Dataset], data_vars: Sequence[str], keys: Iterable[str],
) -> xr.Dataset:
    """
    Selects requested variables in the dataset that are there by default
    (i.e., not added in derived step) and combines the given mapper keys
    into one dataset.

    If all keys are time strings, converts them to time when creating the coordinate.
    """
    keys = sorted(keys)
    try:
        time_coords = [parse_datetime_from_str(key) for key in keys]
    except ValueError:
        time_coords = list(keys)
    ds = xr.concat([mapper[key] for key in keys], pd.Index(time_coords, name=TIME_NAME))
    nonderived_vars = nonderived_variables(data_vars, tuple(ds.data_vars))
    ds = safe.get_variables(ds, nonderived_vars)
    return ds


@curry
def _open_dataset(fs: fsspec.AbstractFileSystem, variable_names, filename):
    return xr.open_dataset(fs.open(filename), engine="h5netcdf")[variable_names]


@batches_functions.register
def batches_from_netcdf(
    path: str, variable_names: Iterable[str]
) -> loaders.typing.Batches:
    """
    Loads a series of netCDF files from the given directory, in alphabetical order.

    Args:
        path: path (local or remote) of a directory of netCDF files
    Returns:
        A sequence of batched data
    """
    fs = vcm.get_fs(path)
    filenames = [fname for fname in sorted(fs.ls(path)) if fname.endswith(".nc")]
    return Map(_open_dataset(fs, variable_names), filenames)


@batches_functions.register
def batches_from_serialized(
    path: str,
    zarr_prefix: str = "phys",
    sample_dims: Sequence[str] = ["savepoint", "rank", "horizontal_dimension"],
    savepoints_per_batch: int = 1,
) -> loaders.typing.Batches:
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
    serialized_seq = SerializedSequence(ds)
    flattened_seq = FlattenDims(serialized_seq, sample_dims)

    if savepoints_per_batch > 1:
        batch_args: Sequence[Union[int, slice]] = [
            slice(start, start + savepoints_per_batch)
            for start in range(0, len(flattened_seq), savepoints_per_batch)
        ]
    else:
        batch_args = list(range(len(flattened_seq)))

    def _load_item(item: Union[int, slice]):
        return flattened_seq[item]

    func_seq = Map(_load_item, batch_args)

    return func_seq
