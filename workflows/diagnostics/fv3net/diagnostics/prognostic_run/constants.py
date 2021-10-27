import numpy as np
from typing import Mapping, Sequence


MovieUrls = Mapping[str, Sequence[str]]

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
    "column_int_dQu",
    "column_int_dQv",
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

GLOBAL_AVERAGE_VARS = GLOBAL_AVERAGE_DYCORE_VARS + GLOBAL_AVERAGE_PHYSICS_VARS

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

GLOBAL_BIAS_VARS = GLOBAL_AVERAGE_DYCORE_VARS + GLOBAL_BIAS_PHYSICS_VARS

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
    "tendency_of_air_temperature_due_to_machine_learning",
    "tendency_of_specific_humidity_due_to_machine_learning",
]

PRECIP_RATE = "total_precip_to_surface"
MASS_STREAMFUNCTION_MID_TROPOSPHERE = "mass_streamfunction_300_700_zonal_and_time_mean"
WVP = "water_vapor_path"
COL_DRYING = "minus_column_integrated_q2"
HISTOGRAM_BINS = {PRECIP_RATE: np.logspace(-1, np.log10(500), 101)}
PERCENTILES = [25, 50, 75, 90, 99, 99.9]
TOP_LEVEL_METRICS = {
    "rmse_5day": ["h500", "tmp850"],
    "rmse_of_time_mean": [PRECIP_RATE, "pwat", "tmp200"],
    "time_and_land_mean_bias": [PRECIP_RATE],
    "rmse_of_time_mean_land": ["tmp850"],
    "tropics_max_minus_min": [MASS_STREAMFUNCTION_MID_TROPOSPHERE],
    "tropical_ascent_region_mean": ["column_integrated_q1"],
}
