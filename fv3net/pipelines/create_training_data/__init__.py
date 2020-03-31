# residuals that the ML is training on
# high resolution tendency - coarse res model's one step tendency
VAR_Q_HEATING_ML = "dQ1"
VAR_Q_MOISTENING_ML = "dQ2"
VAR_Q_U_WIND_ML = "dQU"
VAR_Q_V_WIND_ML = "dQV"

# suffixes denote whether diagnostic variable is from the coarsened
# high resolution prognostic run or the coarse res one step train data run
SUFFIX_HIRES_DIAG = "prog"
SUFFIX_COARSE_TRAIN_DIAG = "train"

RADIATION_VARS = [
    "DSWRFtoa",
    "DSWRFsfc",
    "USWRFtoa",
    "USWRFsfc",
    "DLWRFsfc",
    "ULWRFtoa",
    "ULWRFsfc",
]

RENAMED_HIGH_RES_VARS = {
    **{f"{var}_coarse": f"{var}_prog" for var in RADIATION_VARS},
    **{
        "LHTFLsfc_coarse": "latent_heat_flux_prog",
        "SHTFLsfc_coarse": "sensible_heat_flux_prog",
    },
}

ONE_STEP_VARS = RADIATION_VARS + [
    "total_precipitation",
    "surface_temperature",
    "land_sea_mask",
    "latent_heat_flux",
    "sensible_heat_flux",
    "mean_cos_zenith_angle",
    "surface_geopotential",
    "vertical_thickness_of_atmospheric_layer",
    "vertical_wind",
    "pressure_thickness_of_atmospheric_layer",
    "specific_humidity",
    "air_temperature",
    "x_wind",
    "y_wind",
]

RENAMED_ONE_STEP_VARS = {var: f"{var}_train" for var in RADIATION_VARS}
