import os
import re
from datetime import datetime, timedelta
from typing import Any, Generator, Tuple

import cftime
import fsspec
import numpy as np
import xarray as xr
from dask.delayed import delayed

from vcm.schema_registry import impose_dataset_to_schema
from vcm.combining import combine_array_sequence
from vcm.convenience import open_delayed
from vcm.cubedsphere.constants import FORECAST_TIME_DIM

TIME_FMT = "%Y%m%d.%H%M%S"
SCHEMA_CACHE = {}
RESTART_CATEGORIES = ["fv_core.res", "sfc_data", "fv_tracer", "fv_srf_wnd.res"]
FILE_PREFIX_COORD = "file_prefix"

def open_restarts(url: str, initial_time: str, final_time: str) -> xr.Dataset:
    """Opens all the restart file within a certain path

    The dimension names are the same as the diagnostic output

    Args:
        url: a URL to the root directory of a run directory. Can be any type of protocol
            used by fsspec, such as google cloud storage 'gs://path-to-rundir'. If no
            protocol prefix is used, then it will be assumed to be a path to a local
            file.
        initial_time: A YYYYMMDD.HHMMSS string for the initial condition. The initial
            condition data does not have an time-stamp in its filename, so you must
            provide it using this argument. This only updates the time coordinate of
            the output and does not imply any subselection of time-steps.
        final_time: same as `initial_time` but for the ending time of the simulation.
            Again, the timestamp is not in the filename of the final set of restart
            files.

    Returns:
        a combined dataset of all the restart files. All except the first file of
        each restart-file type (e.g. fv_core.res) will only be lazily loaded. This
        allows opening large datasets out-of-core.

    """
    restart_files = _restart_files_at_url(url, initial_time, final_time)
    arrays = _load_arrays(restart_files)
    return xr.Dataset(combine_array_sequence(arrays, labels=["time", "tile"])).sortby(
        "time"
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


def _set_relative_diff_forecast_time(ds, dt_sec):
    """ Converts the forecast time dim into relative units so that different
    initialization times can be concatenated together

    Args:
        ds: xarray dataset with dim FORECAST_TIME_DIM in absolute time

    Returns:
        dataset with the FORECAST_TIME_DIM converted to relative time after first
        time in dataset (np.timedelta64 [ns])
    """
    num_tsteps = ds.sizes([FILE_PREFIX_COORD])
    return ds.assign_coords(
        {FORECAST_TIME_DIM: [timedelta(seconds=tstep * dt_sec) for tstep in num_tsteps]}
    )


def _parse_forecast_dt(run_dir):
    """

    Args:
        run_dir: run directory assumed to be named with initialization time in TIME_FMT,
         e.g. "20160801.001000"

    Returns:
        float: dt [seconds] as parsed from filenames of first two forecasted restarts
    """
    proto, path = _split_url(run_dir)
    fs = fsspec.filesystem(proto)
    if run_dir[-1] == "/":
        run_dir = run_dir[:-1]
    restart_contents = fs.ls(os.path.join(run_dir, "RESTART"))
    forecast_times = []
    for filename in restart_contents:
        try:
            timestring = _parse_time(os.path.basename(filename))
        except AttributeError:
            timestring = None
        if timestring and timestring not in forecast_times:
            forecast_times.append(timestring)
    forecast_times = sorted(forecast_times)
    t_sample = np.array([_parse_time_string(t) for t in forecast_times[:2]])
    return (t_sample[1]-t_sample[0]).total_seconds()


def _parse_time_string(time):
    t = datetime.strptime(time, TIME_FMT)
    return cftime.DatetimeJulian(t.year, t.month, t.day, t.hour, t.minute, t.second)


def _split_url(url):

    try:
        protocol, path = url.split("://")
    except ValueError:
        protocol = "file"
        path = url

    return protocol, path


def _parse_time(path):
    return re.search(r"(\d\d\d\d\d\d\d\d\.\d\d\d\d\d\d)", path).group(1)


def _get_time(dirname, path, initial_time, final_time):
    if dirname.endswith("INPUT"):
        return initial_time
    elif dirname.endswith("RESTART"):
        try:
            return _parse_time(path)
        except AttributeError:
            return final_time


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


def _parse_first_last_forecast_times(fs, run_dir):
    """
    Args:
        fs: gcsfs GCSFileSystem
        run_dir: run directory assumed to be named with initialization time in TIME_FMT,
         e.g. "20160801.001000"
    Returns:
        strings in TIME_FMT: initialization time and last forecast time
    """
    if run_dir[-1] == "/":
        run_dir = run_dir[:-1]
    restart_contents = fs.ls(os.path.join(run_dir, "RESTART"))
    forecast_times = []
    for filename in restart_contents:
        try:
            timestring = _parse_time(os.path.basename(filename))
        except AttributeError:
            timestring = None
        if timestring and timestring not in forecast_times:
            forecast_times.append(timestring)
    t_init = os.path.basename(run_dir)
    t_last = sorted(forecast_times)[-1]
    return t_init, t_last


def _restart_files_at_url(url, initial_time, final_time):
    """List restart files with a given initial and end time within a particular URL

    Yields:
        (time, restart_category, tile, protocol, path)

    Note:
        the time for the data in INPUT and RESTART cannot be parsed from the file name
        alone so they are required arguments. Some tricky logic such as reading the
        fv_coupler.res file could be done, but I do not think this low-level function
        should have side-effects such as reading a file (which might not always be
        where we expect).

    """
    proto, path = _split_url(url)
    fs = fsspec.filesystem(proto)

    for root, dirs, files in fs.walk(path):
        for file in files:
            path = os.path.join(root, file)
            if _is_restart_file(file):
                time = _get_time(root, file, initial_time, final_time)
                tile = _get_tile(file)
                category = _parse_category(file)
                yield time, category, tile, proto, path


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
    for (time, restart_category, tile, protocol, path) in restart_files:
        ds = _load_restart_lazily(protocol, path, restart_category)
        ds_standard_metadata = standardize_metadata(ds)
        time_obj = _parse_time_string(time)
        for var in ds_standard_metadata:
            yield var, (time_obj, tile), ds_standard_metadata[var]
