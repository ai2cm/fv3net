import xarray as xr
from typing import Tuple

HORIZONTAL_DIMS = ["x", "y", "tile"]

# argument typehint for diags in save_prognostic_run_diags but used
# by multiple modules split out to group operations and simplify
# the diagnostic script
DiagArg = Tuple[xr.Dataset, xr.Dataset, xr.Dataset]

VERIFICATION_CATALOG_ENTRIES = {
    "nudged_shield_40day": {
        "physics": ("40day_c48_gfsphysics_15min_may2020",),
        "dycore": ("40day_c48_atmos_8xdaily_may2020",),
    },
    "nudged_fv3gfs_yearlong": {
        "dycore": ("2016_c48_nudged_fv3gfs_dycore_output",),
        "physics": ("2016_c48_nudged_fv3gfs_physics_output",),
    },
}

GLOBAL_AVERAGE_DYCORE_VARS = [
    "UGRDlowest",
    "UGRD850",
    "UGRD200",
    "TMP500_300",
    "TMPlowest",
    "TMP850",
    "TMP500",
    "TMP200",
    "w500",
    "h500",
    "RH1000",
    "RH850",
    "RH500",
    "PRESsfc",
    "PWAT",
    "VIL",
    "iw",
]

GLOBAL_AVERAGE_PHYSICS_VARS = [
    "column_integrated_pQ1",
    "column_integrated_dQ1",
    "column_integrated_Q1",
    "column_integrated_pQ2",
    "column_integrated_dQ2",
    "column_integrated_Q2",
    "CPRATsfc",
    "PRATEsfc",
    "LHTFLsfc",
    "SHTFLsfc",
    "DSWRFsfc",
    "USWRFsfc",
    "DSWRFtoa",
    "USWRFtoa",
    "ULWRFtoa",
    "ULWRFsfc",
    "DLWRFsfc",
    "TMP2m",
    "TMPsfc",
    "UGRD10m",
    "MAXWIND10m",
    "SOILM",
    "SOILT1",
]

DIURNAL_CYCLE_VARS = [
    "column_integrated_dQ1",
    "column_integrated_pQ1",
    "column_integrated_Q1",
    "column_integrated_dQ2",
    "column_integrated_pQ2",
    "column_integrated_Q2",
    "PRATEsfc",
    "LHTFLsfc",
]
