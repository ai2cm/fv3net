from datetime import datetime, timedelta
import os
import numpy as np
from typing import List, Mapping

import fsspec
import fv3config


# this module assumes that analysis files are at 00Z, 06Z, 12Z and 18Z
SECONDS_IN_HOUR = 60 * 60
NUDGE_HOURS = np.array([0, 6, 12, 18])  # hours at which analysis data is available
NUDGE_INTERVAL = 6  # hours
NUDGE_FILENAME_PATTERN = "%Y%m%d_%HZ_T85LR.nc"
NUDGE_BUCKET = "gs://vcm-ml-data/2019-12-02-year-2016-T85-nudging-data"
NUDGE_FILE_TARGET = "INPUT"  # where to put analysis files in rundir


def _datetime_from_current_date(current_date: List[int]) -> datetime:
    """Return datetime object given current_date, FV3GFS's time structure"""
    year = current_date[0]
    month = current_date[1]
    day = current_date[2]
    hour = current_date[3]
    minute = current_date[4]
    second = current_date[5]
    return datetime(year, month, day, hour, minute, second)


def _get_first_nudge_time(start_time: datetime) -> datetime:
    """Return datetime object for the last nudging time preceding or concurrent
     with start_time"""
    first_nudge_hour = NUDGE_HOURS[np.argmax(NUDGE_HOURS > start_time.hour) - 1]
    return datetime(start_time.year, start_time.month, start_time.day, first_nudge_hour)


def _get_nudge_time_list(config: Mapping) -> List[datetime]:
    """Return list of datetime objects corresponding to times at which analysis files
    are required for nudging for a given model run configuration"""
    current_date = config["namelist"]["coupler_nml"]["current_date"]
    start_time = _datetime_from_current_date(current_date)
    first_nudge_time = _get_first_nudge_time(start_time)
    run_duration = fv3config.get_run_duration(config)
    nudge_duration = run_duration + (start_time - first_nudge_time)
    nudge_duration_hours = int(
        np.ceil(nudge_duration.total_seconds() / SECONDS_IN_HOUR)
    )
    nudging_hours = range(0, nudge_duration_hours + NUDGE_INTERVAL, NUDGE_INTERVAL)
    return [first_nudge_time + timedelta(hours=hour) for hour in nudging_hours]


def _get_nudge_filename_list(config: Mapping) -> List[str]:
    """Return list of filenames of all nudging files required"""
    time_list = _get_nudge_time_list(config)
    return [time.strftime(NUDGE_FILENAME_PATTERN) for time in time_list]


def _get_nudge_files_asset_list(config: Mapping) -> List[Mapping]:
    """Return list of fv3config assets for all nudging files required for a given
    model run configuration"""
    return [
        fv3config.get_asset_dict(NUDGE_BUCKET, file, target_location=NUDGE_FILE_TARGET)
        for file in _get_nudge_filename_list(config)
    ]


def _get_and_write_nudge_files_description_asset(config: Mapping, config_url: str) -> Mapping:
    """Write a text file with list of all nudging files required  (which the
    model requires to know what the nudging files are called) and return an fv3config
    asset pointing to this text file."""
    fname_list_filename = config["namelist"]["fv_nwp_nudge_nml"]["input_fname_list"]
    fname_list_url = os.path.join(config_url, fname_list_filename)
    fname_list_contents = "\n".join(_get_nudge_filename_list(config))
    with fsspec.open(fname_list_url, "w") as f:
        f.write(fname_list_contents)
    return fv3config.get_asset_dict(config_url, fname_list_filename)


def update_config_for_nudging(config: Mapping, config_url: str) -> Mapping:
    """Add assets to config for all nudging files and for the text file listing
    nudging files. This text file will be written to config_url.

    Args:
        config: an fv3config configuration dictionary
        config_url: path where text file describing nudging files will be written.
            File will be written to {config_url}/{input_fname_list} where input_fname_list
            is a namelist parameter in the fv_nwp_nudge_nml namelist of config.

    Returns:
        config dict updated to include all required nudging files
    """
    if "patch_files" not in config:
        config["patch_files"] = []
    config["patch_files"].append(
        _get_and_write_nudge_files_description_asset(config, config_url)
    )
    config["patch_files"].extend(_get_nudge_files_asset_list(config))
    return config
