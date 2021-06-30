import numpy as np
import xarray as xr
from typing import Tuple

HORIZONTAL_DIMS = ["x", "y", "tile"]

# argument typehint for diags in save_prognostic_run_diags but used
# by multiple modules split out to group operations and simplify
# the diagnostic script
DiagArg = Tuple[xr.Dataset, xr.Dataset, xr.Dataset]

RMSE_VARS = [
    "UGRDlowest",
    "VGRDlowest",
    "UGRD850",
    "UGRD200",
    "VGRD850",
    "VGRD200",
    "VORT850",
    "VORT200",
    "TMP500_300",
    "TMPlowest",
    "TMP850",
    "TMP200",
    "h500",
    "PRMSL",
    "PRESsfc",
    "PWAT",
    "VIL",
    "iw",
]

GLOBAL_AVERAGE_DYCORE_VARS = [
    "UGRD850",
    "UGRD200",
    "TMP500_300",
    "TMPlowest",
    "TMP850",
    "TMP200",
    "w500",
    "h500",
    "RH1000",
    "RH850",
    "RH500",
    "q1000",
    "q500",
    "PRMSL",
    "PRESsfc",
    "PWAT",
    "VIL",
    "iw",
]

GLOBAL_AVERAGE_PHYSICS_VARS = [
    "column_integrated_pQ1",
    "column_integrated_dQ1",
    "column_integrated_nQ1",
    "column_integrated_Q1",
    "column_integrated_pQ2",
    "column_integrated_dQ2",
    "column_integrated_nQ2",
    "column_integrated_Q2",
    "vertical_mean_dQu",
    "vertical_mean_dQv",
    "total_precip_to_surface",
    "PRATEsfc",
    "LHTFLsfc",
    "SHTFLsfc",
    "DSWRFsfc",
    "DLWRFsfc",
    "USWRFtoa",
    "ULWRFtoa",
    "TMPsfc",
    "UGRD10m",
    "MAXWIND10m",
    "SOILM",
]

GLOBAL_BIAS_PHYSICS_VARS = [
    "column_integrated_Q1",
    "column_integrated_Q2",
    "total_precip_to_surface",
    "LHTFLsfc",
    "SHTFLsfc",
    "USWRFtoa",
    "ULWRFtoa",
    "TMPsfc",
    "UGRD10m",
    "MAXWIND10m",
    "SOILM",
]

DIURNAL_CYCLE_VARS = [
    "column_integrated_dQ1",
    "column_integrated_nQ1",
    "column_integrated_pQ1",
    "column_integrated_Q1",
    "column_integrated_dQ2",
    "column_integrated_nQ2",
    "column_integrated_pQ2",
    "column_integrated_Q2",
    "total_precip_to_surface",
    "PRATEsfc",
    "LHTFLsfc",
]

TIME_MEAN_VARS = [
    "column_integrated_Q1",
    "column_integrated_Q2",
    "total_precip_to_surface",
    "UGRD850",
    "UGRD200",
    "TMPsfc",
    "TMP850",
    "TMP200",
    "h500",
    "w500",
    "PRESsfc",
    "PWAT",
    "LHTFLsfc",
    "SHTFLsfc",
    "DSWRFsfc",
    "DLWRFsfc",
    "USWRFtoa",
    "ULWRFtoa",
]

PRESSURE_INTERPOLATED_VARS = [
    "air_temperature",
    "specific_humidity",
    "eastward_wind",
    "northward_wind",
    "vertical_wind",
    "dQ1",
    "dQ2",
    "dQu",
    "dQv",
]

PRECIP_RATE = "total_precip_to_surface"
HISTOGRAM_BINS = {PRECIP_RATE: np.logspace(-1, np.log10(500), 101)}
PERCENTILES = [25, 50, 75, 90, 99, 99.9]
TOP_LEVEL_METRICS = {
    "rmse_5day": ["h500", "tmp850"],
    "rmse_of_time_mean": [PRECIP_RATE, "pwat", "tmp200"],
    "time_and_land_mean_bias": [PRECIP_RATE],
    "rmse_of_time_mean_land": ["tmp850"],
    "tropics_max_minus_min": ["psi_300_to_700"],
    "tropical_ascent_region_mean": ["column_integrated_q1"],
}
