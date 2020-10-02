import xarray as xr
from typing import Tuple

HORIZONTAL_DIMS = ["x", "y", "tile"]

# argument typehint for diags in save_prognostic_run_diags but used
# by multiple modules split out to group operations and simplify
# the diagnostic script
DiagArg = Tuple[xr.Dataset, xr.Dataset, xr.Dataset]

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
    "q1000",
    "q850",
    "q500",
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
    "total_precip",
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

GLOBAL_BIAS_PHYSICS_VARS = [
    "column_integrated_Q1",
    "column_integrated_Q2",
    "total_precip",
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

TIME_MEAN_VARS = [
    "column_integrated_Q1",
    "column_integrated_Q2",
    "total_precip",
]
