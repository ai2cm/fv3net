from datetime import datetime, timedelta
import os
import numpy as np
from typing import List, Mapping

import fsspec
import fv3config

# this module assumes that analysis files are at 00Z, 06Z, 12Z and 18Z
SECONDS_IN_HOUR = 60 * 60
NUDGE_HOURS = np.array([0, 6, 12, 18])  # hours at which analysis data is available
NUDGE_FILE_TARGET = "INPUT"  # where to put analysis files in rundir


def _most_recent_nudge_time(start_time: datetime) -> datetime:
    """Return datetime object for the last nudging time preceding or concurrent
     with start_time"""
    first_nudge_hour = _most_recent_hour(start_time.hour)
    return datetime(start_time.year, start_time.month, start_time.day, first_nudge_hour)


def _most_recent_hour(current_hour, hour_array=NUDGE_HOURS) -> int:
    """Return latest hour in hour_array that precedes or is concurrent with
    current_hour"""
    first_nudge_hour = hour_array[np.argmax(hour_array > current_hour) - 1]
    return first_nudge_hour


def _get_nudge_time_list(config: Mapping) -> List[datetime]:
    """Return list of datetime objects corresponding to times at which analysis files
    are required for nudging for a given model run configuration"""
    current_date = config["namelist"]["coupler_nml"]["current_date"]
    start_time = datetime(*current_date)
    first_nudge_time = _most_recent_nudge_time(start_time)
    run_duration = fv3config.get_run_duration(config)
    nudge_duration = run_duration + (start_time - first_nudge_time)
    nudge_duration_hours = int(
        np.ceil(nudge_duration.total_seconds() / SECONDS_IN_HOUR)
    )
    nudge_interval = NUDGE_HOURS[1] - NUDGE_HOURS[0]
    nudging_hours = range(0, nudge_duration_hours + nudge_interval, nudge_interval)
    return [first_nudge_time + timedelta(hours=hour) for hour in nudging_hours]


def _get_nudge_filename_list(config: Mapping) -> List[str]:
    """Return list of filenames of all nudging files required"""
    nudge_filename_pattern = config["gfs_analysis_data"]["filename_pattern"]
    time_list = _get_nudge_time_list(config)
    return [time.strftime(nudge_filename_pattern) for time in time_list]


def _get_nudge_files_asset_list(config: Mapping) -> List[Mapping]:
    """Return list of fv3config assets for all nudging files required for a given
    model run configuration"""
    nudge_url = config["gfs_analysis_data"]["url"]
    return [
        fv3config.get_asset_dict(nudge_url, file, target_location=NUDGE_FILE_TARGET)
        for file in _get_nudge_filename_list(config)
    ]


def _get_nudge_files_description_asset(config: Mapping, config_url: str) -> Mapping:
    """Return an fv3config asset pointing to the text file that the
    model requires to describe the list of nudging files."""
    fname_list_filename = config["namelist"]["fv_nwp_nudge_nml"]["input_fname_list"]
    return fv3config.get_asset_dict(config_url, fname_list_filename)


def _get_input_fname_list_asset(config: Mapping, filename: str) -> Mapping:
    fname_list_contents = "\n".join(_get_nudge_filename_list(config))
    data = fname_list_contents.encode()
    return fv3config.get_bytes_asset_dict(
        data, target_location="", target_name=filename
    )


def enable_nudge_to_observations(config: Mapping) -> Mapping:
    """Return an configuration dictionary with nudging to observations enabled

    Accepts and returns an fv3config dictionary

    Note:
        This function appends to patch_files and alters the namelist
    """
    input_fname_list = "nudging_file_list"
    config = _assoc_nudging_namelist_options(config, input_fname_list=input_fname_list)
    fname_list_asset = _get_input_fname_list_asset(config, input_fname_list)

    patch_files = config.setdefault("patch_files", [])
    patch_files.append(fname_list_asset)
    patch_files.append(_get_nudge_files_asset_list(config))

    return config


def _assoc_nudging_namelist_options(
    config,
    gfs_analysis_url="gs://vcm-ml-data/2019-12-02-year-2016-T85-nudging-data",
    input_fname_list="nudging_file_list",
    tau_ps=21600.0,
    tau_virt=21600.0,
    tau_winds=21600.0,
    tau_q=21600.0,
) -> Mapping:
    """assoc is a common name for adding new items to a dictionary without mutation"""

    # TODO Oli, please indicate which options below are unrelated to nudging
    namelist_overlay = {
        "gfs_analysis_data": {
            "url": gfs_analysis_url,
            "filename_pattern": "%Y%m%d_%HZ_T85LR.nc",
        },
        "namelist": {
            "atmos_model_nml": {"fhout": 2.0, "fhmax": 10000},
            "fv_core_nml": {"nudge": True},
            "gfs_physics_nml": {"fhzero": 2.0, "use_analysis_sst": True},
            "fv_nwp_nudge_nml": {
                "add_bg_wind": False,
                "do_ps_bias": False,
                "ibtrack": True,
                "input_fname_list": input_fname_list,
                "k_breed": 10,
                "kbot_winds": 0,
                "mask_fac": 0.2,
                "nf_ps": 3,
                "nf_t": 3,
                "nudge_debug": True,
                "nudge_hght": False,
                "nudge_ps": True,
                "nudge_virt": True,
                "nudge_winds": True,
                "nudge_q": True,
                "r_hi": 5.0,
                "r_lo": 3.0,
                "r_min": 225000.0,
                "t_is_tv": False,
                "tau_ps": tau_ps,
                "tau_virt": tau_virt,
                "tau_winds": tau_winds,
                "tau_q": tau_q,
                "tc_mask": True,
                "time_varying": False,
                "track_file_name": "No_File_specified",
                "use_high_top": True,
            },
        },
    }

    fv3config_overlay = {"namelist": namelist_overlay}
    return vcm.update_nested_dict(config, fv3config_overlay)
