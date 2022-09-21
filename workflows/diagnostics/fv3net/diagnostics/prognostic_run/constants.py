from typing import Mapping, Sequence


MovieUrls = Mapping[str, Sequence[str]]

# Added to atmos diags by fortran model if interval-averaged.
# Removed before coarsening run data to avoid errors.
FORTRAN_TILE_ONLY_VARS = ("average_T1", "average_T2", "average_DT")

# Renamed keys already have "_coarse" suffix removed prior to the renaming being applied
VERIFICATION_RENAME_MAP = {
    "40day_may2020": {"2d": {"TB": "TMPlowest", "tsfc": "TMPsfc"}},
    "1yr_pire_postspinup": {"2d": {"TB": "TMPlowest", "tsfc": "TMPsfc"}},
}

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
    "TMP500",
    "TMP200",
    "h500",
    "q500",
    "PRMSL",
    "PRESsfc",
    "PWAT",
    "water_vapor_path",
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
    "RH200",
    "q1000",
    "q500",
    "q200",
    "PRMSL",
    "PRESsfc",
    "PWAT",
    "water_vapor_path",
    "VIL",
    "iw",
    "VGRD850",
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
    "VGRD10m",
    "is_novelty",
    "centered_score",
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
    "VGRD10m",
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
    "water_vapor_path",
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
    "relative_humidity",
    "eastward_wind",
    "northward_wind",
    "vertical_wind",
    "dQ1",
    "dQ2",
    "dQu",
    "dQv",
    "tendency_of_air_temperature_due_to_machine_learning",
    "tendency_of_specific_humidity_due_to_machine_learning",
    "air_temperature_tendency_due_to_nudging",
    "specific_humidity_tendency_due_to_nudging",
]

PRECIP_RATE = "total_precip_to_surface"
MASS_STREAMFUNCTION_MID_TROPOSPHERE = (
    "mass_streamfunction_300_700_time_dependent_zonal_mean"
)
MASS_STREAMFUNCTION_MID_TROPOSPHERE_TIME_MEAN = (
    "mass_streamfunction_300_700_zonal_and_time_mean"
)

PERCENTILES = [25, 50, 75, 90, 99, 99.9]
TOP_LEVEL_METRICS = {
    "rmse_5day": ["h500", "tmp850"],
    "rmse_of_time_mean": [PRECIP_RATE, "pwat", "tmp200"],
    "time_and_land_mean_bias": [PRECIP_RATE],
    "rmse_of_time_mean_land": ["tmp850"],
    "tropics_max_minus_min": [MASS_STREAMFUNCTION_MID_TROPOSPHERE_TIME_MEAN],
    "tropical_ascent_region_mean": ["column_integrated_q1", "water_vapor_path"],
}

GRID_VARS = ("lat", "latb", "lon", "lonb")
GRID_INTERFACE_COORDS = ("x_interface", "y_interface")

# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.figure.html
DEFAULT_FIGURE_WIDTH = 6.4
DEFAULT_FIGURE_HEIGHT = 4.8

# For inferring a readable aspect ratio for tropical Hovmoller plots
REFERENCE_HOVMOLLER_DURATION_SECONDS = 40 * 86400
