from typing import Any, Generator, Tuple

import xarray as xr
from dask.delayed import delayed

from vcm.combining import combine_array_sequence
from vcm.convenience import open_delayed
from vcm.schema_registry import impose_dataset_to_schema
from vcm.cloud.fsspec import get_fs


from . import _rundir

SCHEMA_CACHE = {}


def open_restarts(url: str) -> xr.Dataset:
    """Opens all the restart file within a certain path

    The dimension names are the same as the diagnostic output

    Args:
        url (str): a URL to the root directory of a run directory.
            Can be any type of protocol used by fsspec, such as google cloud storage
            'gs://path-to-rundir'. If no protocol prefix is used, then it will be
            assumed to be a path to a local file.

    Returns:
        ds (xr.Dataset): a combined dataset of all the restart files. All except
            the first file of each restart-file type (e.g. fv_core.res) will only
            be lazily loaded. This allows opening large datasets out-of-core.

    """
    fs = get_fs(url)
    walker = fs.walk(url)
    restart_files = _rundir.yield_restart_files(walker)
    arrays = _load_arrays(fs, restart_files)
    return xr.Dataset(combine_array_sequence(arrays, labels=["file_prefix", "tile"]))


def open_restarts_with_time_coordinates(url: str) -> xr.Dataset:
    """Opens all the restart file within a certain path, with time coordinates

    The dimension names are the same as the diagnostic output

    Args:
        url (str): a URL to the root directory of a run directory.
            Can be any type of protocol used by fsspec, such as google cloud storage
            'gs://path-to-rundir'. If no protocol prefix is used, then it will be
            assumed to be a path to a local file.

    Returns:
        ds (xr.Dataset): a combined dataset of all the restart files. All except
            the first file of each restart-file type (e.g. fv_core.res) will only
            be lazily loaded. This allows opening large datasets out-of-core.
            Time coordinates are inferred from the run directory's namelist and
            other files.

    """
    file_prefix_dim = "file_prefix"
    time_dim = "time"

    fs = get_fs(url)
    ds = open_restarts(url)
    mapping = _rundir.get_prefix_time_mapping(fs, url)
    ds_with_times = _replace_1d_coord_by_mapping(ds, mapping, file_prefix_dim, time_dim)
    return ds_with_times.sortby(time_dim)


def standardize_metadata(ds: xr.Dataset) -> xr.Dataset:
    """Update the meta-data of an individual restart file

    This drops the singleton time dimension and applies the known dimensions
    listed in `vcm.schema` and `vcm._schema_registry`.
    """
    try:
        ds_no_time = ds.isel(Time=0).drop("Time")
    except ValueError:
        ds_no_time = ds
    return impose_dataset_to_schema(ds_no_time)


def _replace_1d_coord_by_mapping(ds, mapping, old_dim, new_dim="time"):
    coord = ds[old_dim]
    times = xr.DataArray([mapping[prefix.item()] for prefix in coord], dims=[old_dim])
    return ds.assign_coords({new_dim: times}).swap_dims({old_dim: new_dim})


def _load_restart(fs, path):
    with fs.open(path) as f:
        return xr.open_dataset(f).compute()


def _load_restart_with_schema(fs, path, schema):
    promise = delayed(_load_restart)(fs, path)
    return open_delayed(promise, schema)


def _load_restart_lazily(fs, path, restart_category):
    # only actively load the initial data
    if restart_category in SCHEMA_CACHE:
        schema = SCHEMA_CACHE[restart_category]
    else:
        schema = _load_restart(fs, path)
        SCHEMA_CACHE[restart_category] = schema

    return _load_restart_with_schema(fs, path, schema)


def _load_arrays(
    fs,
    restart_files,
) -> Generator[Tuple[Any, Tuple, xr.DataArray], None, None]:
    # use the same schema for all coupler_res
    for (file_prefix, restart_category, tile, path) in restart_files:
        ds = _load_restart_lazily(fs, path, restart_category)
        ds_standard_metadata = standardize_metadata(ds)
        for var in ds_standard_metadata:
            yield var, (file_prefix, tile), ds_standard_metadata[var]
