"""Routines for reading metadata from the path names and input namelist of a
FV3 run directory.

Please no xarray in this module.

"""
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence, Mapping

import cftime
import fsspec
import pandas as pd
from toolz import memoize

import f90nml
from vcm.convenience import parse_timestep_str_from_path
from vcm.cubedsphere.constants import RESTART_CATEGORIES

__all__ = ["get_prefix_time_mapping", "restart_files_at_url"]


def split(url):
    path = Path(url)
    no_timestamp = path.name.endswith("INPUT") or path.name.endswith("RESTART")

    if no_timestamp:
        parent = path.parent
    else:
        parent = path.parent.parent

    return str(parent), str(path.relative_to(parent))


def append_if_not_present(list, item):
    if item not in list:
        list.append(item)


def _parse_time(name):
    return re.search(r"(\d\d\d\d\d\d\d\d.\d\d\d\d\d\d)", name).group(1)


def get_prefixes(fs, url):
    prefixes = ["INPUT"]
    restarts = fs.glob(url + "/RESTART/????????.??????.*")
    for restart in restarts:
        time = _parse_time(Path(restart).name)
        append_if_not_present(prefixes, os.path.join("RESTART", time))
    prefixes.append("RESTART")
    return prefixes


@memoize(key=lambda args, kwargs: args[1])
def get_prefix_time_mapping(
    fs: fsspec.AbstractFileSystem, url: str
) -> Mapping[str, cftime.DatetimeJulian]:
    """Return a dictionary mapping restart file prefixes to times

    Args:
        fs: fsspec filesystem object
        url: url to the run-directory
    Returns:
        a mapping from "file_prefix (e.g. "INPUT" or "RESTART/YYYYMMDD.HHMMSS"
        timestamp) to parsed date time objects

    """
    times = get_restart_times(fs, url)
    prefixes = get_prefixes(fs, url)
    return dict(zip(prefixes, times))


def sorted_file_prefixes(prefixes):

    return sorted(prefix for prefix in prefixes if prefix not in ["INPUT", "RESTART"])


def _get_initial_time(prefix):
    return str(Path(prefix).parent.parent.name)


def get_restart_times(fs, url: str) -> Sequence[cftime.DatetimeJulian]:
    """Reads the run directory's files to infer restart forecast times

    Due to the challenges of directly parsing the forecast times from the restart files,
    it is more robust to read the ime outputs from the namelist and coupler.res
    in the run directory. This function implements that ability.

    Args:
        fs: fsspec filesystem object
        url (str): a URL to the root directory of a run directory.
            Can be any type of protocol used by fsspec, such as google cloud storage
            'gs://path-to-rundir'. If no protocol prefix is used, then it will be
            assumed to be a path to a local file.

    Returns:
        time Sequence[cftime.DatetimeJulian]: a list of time coordinates
    """
    namelist_path = _get_namelist_path(fs, url)
    config = _config_from_fs_namelist(fs, namelist_path)
    initialization_time = _get_current_date(config, fs, url)
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
        return "INPUT"
    elif dirname.endswith("RESTART"):
        try:
            return os.path.join("RESTART", parse_timestep_str_from_path(path))
        except ValueError:
            return "RESTART"


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


def restart_files_at_url(url):
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


def _get_namelist_path(fs, url):
    for root, dirs, files in fs.walk(url):
        for file in files:
            if _is_namelist_file(file):
                return os.path.join(root, file)


def _is_namelist_file(file):
    return "input.nml" in file


def _get_coupler_res_path(fs, url):
    for root, dirs, files in fs.walk(url):
        for file in files:
            if _is_coupler_res_file(root, file):
                return os.path.join(root, file)


def _is_coupler_res_file(root, file):
    return "INPUT/coupler.res" in os.path.join(root, file)


def _config_from_fs_namelist(fs, namelist_path):
    with fs.open(namelist_path, "rt") as f:
        return _to_nested_dict(f90nml.read(f).items())


def _to_nested_dict(source):
    return_value = dict(source)
    for name, value in return_value.items():
        if isinstance(value, f90nml.Namelist):
            return_value[name] = _to_nested_dict(value)
    return return_value


def _get_current_date(config, fs, url):
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
            coupler_res_filename = _get_coupler_res_path(fs, url)
            current_date = _get_current_date_from_coupler_res(fs, coupler_res_filename)
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


def _get_current_date_from_coupler_res(fs, coupler_res_filename):
    """Return a timedelta indicating the duration of the run.
    Note: Mostly copied from fv3config, but with fsspec capabilities added
    """
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
