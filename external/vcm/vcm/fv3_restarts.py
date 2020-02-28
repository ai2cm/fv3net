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
from vcm.convenience import open_delayed
from vcm.cubedsphere.constants import RESTART_CATEGORIES
from vcm import parse_timestep_from_path


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


def get_restart_times(url: str) -> Sequence[cftime.DatetimeJulian]:
    """Reads the run directory's files to infer restart forecast times

    Due to the challenges of directly parsing the forecast times from the restart files,
    it is more robust to read the ime outputs from the namelist and coupler.res
    in the run directory. This function implements that ability.

    Args:
        url (str): a URL to the root directory of a run directory.
            Can be any type of protocol used by fsspec, such as google cloud storage
            'gs://path-to-rundir'. If no protocol prefix is used, then it will be
            assumed to be a path to a local file.

    Returns:
        time Sequence[cftime.DatetimeJulian]: a list of time coordinates
    """
    proto, namelist_path = _get_namelist_path(url)
    config = _config_from_fs_namelist(proto, namelist_path)
    initialization_time = _get_current_date(config, url)
    duration = _get_run_duration(config)
    interval = _get_restart_interval(config)
    forecast_time = _get_forecast_time_index(initialization_time, duration, interval)
    return forecast_time


def _split_url(url):

    try:
        protocol, path = url.split("://")
    except ValueError:
        protocol = "file"
        path = url

    return protocol, path


def _get_file_prefix(dirname, path):
    if dirname.endswith("INPUT"):
        return "INPUT/"
    elif dirname.endswith("RESTART"):
        try:
            return os.path.join("RESTART", parse_timestep_from_path(path))
        except AttributeError:
            return "RESTART/"


def _sort_file_prefixes(ds, url):

    if "INPUT/" not in ds.file_prefix:
        raise ValueError(
            "Open restarts did not find the input set "
            f"of restart files for run directory {url}."
        )
    if "RESTART/" not in ds.file_prefix:
        raise ValueError(
            "Open restarts did not find the final set "
            f"of restart files for run directory {url}."
        )

    intermediate_prefixes = sorted(
        [
            prefix.item()
            for prefix in ds.file_prefix
            if prefix.item() not in ["INPUT/", "RESTART/"]
        ]
    )

    return xr.concat(
        [
            ds.sel(file_prefix="INPUT/"),
            ds.sel(file_prefix=intermediate_prefixes),
            ds.sel(file_prefix="RESTART/"),
        ],
        dim="file_prefix",
    )


def _parse_category(path):
    cats_in_path = {category for category in RESTART_CATEGORIES if category in path}
    if len(cats_in_path) == 1:
        return cats_in_path.pop()
    else:
        # Check that the file only matches one restart category for safety
        # it not clear if this is completely necessary, but it ensures the output of
        # this routine is more predictable
        raise ValueError("Multiple categories present in filename.")


def _get_tile(path):
    """Get tile number

    Following python, but unlike FV3, the first tile number is 0. In other words, the
    tile number of `.tile1.nc` is 0.

    This avoids confusion when using the outputs of :ref:`open_restarts`.
    """
    tile = re.search(r"tile(\d)\.nc", path).group(1)
    return int(tile) - 1


def _is_restart_file(path):
    return any(category in path for category in RESTART_CATEGORIES) and "tile" in path


def _restart_files_at_url(url):
    """List restart files with a given initial and end time within a particular URL

    Yields:
        (time, restart_category, tile, protocol, path)

    """
    proto, path = _split_url(url)
    fs = fsspec.filesystem(proto)

    for root, dirs, files in fs.walk(path):
        for file in files:
            path = os.path.join(root, file)
            if _is_restart_file(file):
                file_prefix = _get_file_prefix(root, file)
                tile = _get_tile(file)
                category = _parse_category(file)
                yield file_prefix, category, tile, proto, path


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


def _get_namelist_path(url):

    proto, path = _split_url(url)
    fs = fsspec.filesystem(proto)

    for root, dirs, files in fs.walk(path):
        for file in files:
            if _is_namelist_file(file):
                return proto, os.path.join(root, file)


def _is_namelist_file(file):
    return "input.nml" in file


def _get_coupler_res_path(url):

    proto, path = _split_url(url)
    fs = fsspec.filesystem(proto)

    for root, dirs, files in fs.walk(path):
        for file in files:
            if _is_coupler_res_file(root, file):
                return proto, os.path.join(root, file)


def _is_coupler_res_file(root, file):
    return "INPUT/coupler.res" in os.path.join(root, file)


def _config_from_fs_namelist(proto, namelist_path):
    fs = fsspec.filesystem(proto)
    with fs.open(namelist_path, "rt") as f:
        return _to_nested_dict(f90nml.read(f).items())


def _to_nested_dict(source):
    return_value = dict(source)
    for name, value in return_value.items():
        if isinstance(value, f90nml.Namelist):
            return_value[name] = _to_nested_dict(value)
    return return_value


def _get_current_date(config, url):
    """Return current_date as a datetime from configuration dictionary
    Note: Mostly copied from fv3config, but with fsspec capabilities added
    """
    force_date_from_namelist = config["coupler_nml"].get(
        "force_date_from_namelist", False
    )
    # following code replicates the logic that the fv3gfs model
    # uses to determine the current_date
    if force_date_from_namelist:
        current_date = config["coupler_nml"].get("current_date", [0, 0, 0, 0, 0, 0])
    else:
        try:
            proto, coupler_res_filename = _get_coupler_res_path(url)
            current_date = _get_current_date_from_coupler_res(
                proto, coupler_res_filename
            )
        except TypeError:
            current_date = config["coupler_nml"].get("current_date", [0, 0, 0, 0, 0, 0])
    return datetime(
        **{
            time_unit: value
            for time_unit, value in zip(
                ("year", "month", "day", "hour", "minute", "second"), current_date
            )
        }
    )


def _get_current_date_from_coupler_res(proto, coupler_res_filename):
    """Return a timedelta indicating the duration of the run.
    Note: Mostly copied from fv3config, but with fsspec capabilities added
    """
    fs = fsspec.filesystem(proto)
    with fs.open(coupler_res_filename, "rt") as f:
        third_line = f.readlines()[2]
        current_date = [int(d) for d in re.findall(r"\d+", third_line)]
        if len(current_date) != 6:
            raise ValueError(
                f"{coupler_res_filename} does not have a valid current model time"
                "(need six integers on third line)"
            )
    return current_date


def _get_run_duration(config):
    """Return a timedelta indicating the duration of the run.
    Note: Mostly copied from fv3config
    """
    coupler_nml = config.get("coupler_nml", {})
    months = coupler_nml.get("months", 0)
    if months != 0:  # months have no set duration and thus cannot be timedelta
        raise ValueError(f"namelist contains non-zero value {months} for months")
    return timedelta(
        **{
            name: coupler_nml.get(name, 0)
            for name in ("seconds", "minutes", "hours", "days")
        }
    )


def _get_restart_interval(config):
    config = config["coupler_nml"]
    return timedelta(
        seconds=(config.get("restart_secs", 0) + 86400 * config.get("restart_days", 0))
    )


def _get_forecast_time_index(initialization_time, duration, interval):
    """Return a list of cftime.DatetimeJulian objects for the restart output
    """
    if interval == timedelta(seconds=0):
        interval = duration
    end_time = initialization_time + duration
    return [
        cftime.DatetimeJulian(
            timestamp.year,
            timestamp.month,
            timestamp.day,
            timestamp.hour,
            timestamp.minute,
            timestamp.second,
        )
        for timestamp in pd.date_range(
            start=initialization_time, end=end_time, freq=interval
        )
    ]
