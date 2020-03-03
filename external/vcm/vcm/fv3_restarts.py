import os
import re
from datetime import datetime, timedelta
from typing import Any, Generator, Tuple, Sequence

import cftime
import fsspec
import xarray as xr
import pandas as pd
from dask.delayed import delayed
import f90nml

from vcm.schema_registry import impose_dataset_to_schema
from vcm.combining import combine_array_sequence
from vcm.convenience import open_delayed, parse_timestep_str_from_path
from vcm.cubedsphere.constants import RESTART_CATEGORIES

from _rundir import get_restart_times, _sort_file_prefixes, _restart_files_at_url

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
    restart_files = _restart_files_at_url(url)
    arrays = _load_arrays(restart_files)
    return _sort_file_prefixes(
        xr.Dataset(combine_array_sequence(arrays, labels=["file_prefix", "tile"])), url
    )


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
    ds = open_restarts(url)
    try:
        times = get_restart_times(url)
    except (ValueError, TypeError) as e:
        print(
            f"Warning, inferring time dimensions failed: {e}.\n"
            f"Returning no time coordinates for run directory at {url}."
        )
        return ds
    else:
        return ds.assign_coords({"time": ("file_prefix", times)}).swap_dims(
            {"file_prefix": "time"}
        )


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


def _load_restart(protocol, path):
    fs = fsspec.filesystem(protocol)
    with fs.open(path) as f:
        return xr.open_dataset(f).compute()


def _load_restart_with_schema(protocol, path, schema):
    promise = delayed(_load_restart)(protocol, path)
    return open_delayed(promise, schema)


def _load_restart_lazily(protocol, path, restart_category):
    # only actively load the initial data
    if restart_category in SCHEMA_CACHE:
        schema = SCHEMA_CACHE[restart_category]
    else:
        schema = _load_restart(protocol, path)
        SCHEMA_CACHE[restart_category] = schema

    return _load_restart_with_schema(protocol, path, schema)


def _load_arrays(
    restart_files,
) -> Generator[Tuple[Any, Tuple, xr.DataArray], None, None]:
    # use the same schema for all coupler_res
    for (file_prefix, restart_category, tile, protocol, path) in restart_files:
        ds = _load_restart_lazily(protocol, path, restart_category)
        ds_standard_metadata = standardize_metadata(ds)
        for var in ds_standard_metadata:
            yield var, (file_prefix, tile), ds_standard_metadata[var]