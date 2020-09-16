from datetime import datetime, timedelta
import os
import numpy as np
from typing import List, Mapping, Sequence

import fv3config


def enable_nudge_to_observations(
    duration: timedelta,
    current_date: Sequence[int],
    nudge_filename_pattern: str = "%Y%m%d_%HZ_T85LR.nc",
    nudge_url: str = "gs://vcm-ml-data/2019-12-02-year-2016-T85-nudging-data",
    copy_method: str = "copy",
) -> Mapping:
    """Return config overlay for a nudged to observation run

    Args:
        duration: fv3gfs run duration
        current_date: start time of run as sequence of 6 integers
        nudge_filename_pattern: naming convention for GFS analysis files
        nudge_url: location of GFS analysis files
        copy_method: fv3config asset copy_method for analysis files

    Returns:
        fv3config overlay with default nudging options and assets for analysis files


    This sets background namelist options and adds the necessary analysis
    data files to the patch_files. To actually include nudging, the user must
    enable the nudging for each field and set the coresponding timescale.
    
    For example, this can be done using the following user configuration::
  
        namelist:
            fv_nwp_nudge_nml:
                nudge_hght: false
                nudge_ps: true
                nudge_virt: true
                nudge_winds: true
                nudge_q: true
                tau_ps: 21600.0
                tau_virt: 21600.0
                tau_winds: 21600.0
                tau_q: 21600.0
    """

    nudging_assets = fv3config.get_nudging_assets(
        duration,
        current_date,
        nudge_url,
        nudge_filename_pattern=nudge_filename_pattern,
        copy_method=copy_method,
    )

    file_names = [
        os.path.join(asset["target_location"], asset["target_name"])
        for asset in nudging_assets
    ]

    return {
        "gfs_analysis_data": {
            "url": nudge_url,
            "filename_pattern": nudge_filename_pattern,
            "copy_method": copy_method,
        },
        "namelist": _namelist(file_names),
        "patch_files": nudging_assets,
    }


def _namelist(file_names: Sequence[str]) -> Mapping:
    return {
        "fv_core_nml": {"nudge": True},
        "gfs_physics_nml": {"use_analysis_sst": True},
        "fv_nwp_nudge_nml": {
            "add_bg_wind": False,
            "do_ps_bias": False,
            "file_names": file_names,
            "ibtrack": True,
            "k_breed": 10,
            "kbot_winds": 0,
            "mask_fac": 0.2,
            "nf_ps": 3,
            "nf_t": 3,
            "nudge_debug": True,
            "r_hi": 5.0,
            "r_lo": 3.0,
            "r_min": 225000.0,
            "t_is_tv": False,
            "tc_mask": True,
            "time_varying": False,
            "track_file_name": "No_File_specified",
            "use_high_top": True,
        },
    }
