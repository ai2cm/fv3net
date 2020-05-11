INIT_TIME_DIM = "initial_time"
FORECAST_TIME_DIM = "forecast_time"
DELTA_DIM = "model_run"
OUTPUT_NC_FILENAME = "one_step_diag_data.nc"
ONE_STEP_ZARR = "big.zarr"
ZARR_STEP_DIM = "step"
ZARR_STEP_NAMES = {"begin": "begin", "after_physics": "after_physics"}
REPORT_TITLE = "One-step diagnostics"
FIGURE_METADATA_FILE = "figure_metadata.yml"
METADATA_TABLE_FILE = "metadata_table.yml"

SFC_VARIABLES = [
    "DSWRFtoa",
    "DSWRFsfc",
    "USWRFtoa",
    "USWRFsfc",
    "DLWRFsfc",
    "ULWRFtoa",
    "ULWRFsfc",
]

VARS_FROM_ZARR = [
    "specific_humidity",
    "cloud_ice_mixing_ratio",
    "cloud_water_mixing_ratio",
    "rain_mixing_ratio",
    "snow_mixing_ratio",
    "graupel_mixing_ratio",
    "vertical_wind",
    "air_temperature",
    "pressure_thickness_of_atmospheric_layer",
    "latent_heat_flux",
    "sensible_heat_flux",
    "total_precipitation",
] + SFC_VARIABLES

GRID_VARS = ["lat", "lon", "latb", "lonb", "area", "land_sea_mask"]

HI_RES_DIAGS_MAPPING = {name: name for name in SFC_VARIABLES}
HI_RES_DIAGS_MAPPING.update(
    {
        "latent_heat_flux": "LHTFLsfc",
        "sensible_heat_flux": "SHTFLsfc",
        "total_precipitation": "PRATEsfc",
    }
)

HI_RES_DIAGS_DIMS = {"grid_xt": "x", "grid_yt": "y"}

ABS_VARS = [
    "surface_pressure",
    "liquid_water_equivalent",
    "column_integrated_heat",
    "vertical_wind",
]

LEVEL_VARS = [
    ("vertical_wind", 40),
    ("vertical_wind_abs", 40),
]

GLOBAL_MEAN_2D_VARS = {
    "surface_pressure_abs": {"var_type": ["tendencies"], "scale": [0.12]},
    "liquid_water_equivalent_abs": {"var_type": ["tendencies"], "scale": [0.0006]},
    "liquid_water_equivalent": {"var_type": ["tendencies"], "scale": [0.0006]},
    "column_integrated_heat": {
        "var_type": ["tendencies", "states"],
        "scale": [1000, None],
    },
    "column_integrated_heat_abs": {"var_type": ["tendencies"], "scale": [2500]},
    "vertical_wind_abs_level_40": {"var_type": ["states"], "scale": [0.05]},
    "latent_heat_flux": {"var_type": ["states"], "scale": [None]},
    "sensible_heat_flux": {"var_type": ["states"], "scale": [None]},
    "total_precipitation": {"var_type": ["states"], "scale": [None]},
}

DIURNAL_VAR_MAPPING = {
    "net_heating_diurnal": {
        "hi-res - coarse": {
            "name": "column_integrated_heating",
            "var_type": "tendencies",
        },
        "physics": {"name": "net_heating_physics", "var_type": "states"},
        "scale": 500,
    },
    "net_precipitation_diurnal": {
        "hi-res - coarse": {
            "name": "minus_column_integrated_moistening",
            "var_type": "tendencies",
        },
        "physics": {"name": "net_precipitation_physics", "var_type": "states"},
        "scale": 10,
    },
    "total_precipitation_diurnal": {
        "hi-res - coarse": {"name": "total_precipitation", "var_type": "states"},
        "physics": {"name": "total_precipitation", "var_type": "states"},
        "scale": 10,
    },
    "evaporation_diurnal": {
        "hi-res - coarse": {"name": "evaporation", "var_type": "states"},
        "physics": {"name": "evaporation", "var_type": "states"},
        "scale": 10,
    },
    "vertical_wind_diurnal": {
        "hi-res - coarse": {"name": "vertical_wind_level_40", "var_type": "states"},
        "physics": {"name": "vertical_wind_level_40", "var_type": "states"},
        "scale": 0.05,
    },
}

DQ_MAPPING = {
    "Q1": {
        "physics_name": "net_heating",
        "tendency_diff_name": "column_integrated_heating",
        "scale": 1000,
    },
    "Q2": {
        "physics_name": "net_precipitation",
        "tendency_diff_name": "minus_column_integrated_moistening",
        "scale": 10,
    },
}

DQ_PROFILE_MAPPING = {
    "air_temperature": {"name": "dQ1", "var_type": "tendencies", "scale": 5e-5},
    "specific_humidity": {"name": "dQ2", "var_type": "tendencies", "scale": 1e-7},
    "vertical_wind": {"name": "dW", "var_type": "states", "scale": 0.025},
}

PROFILE_COMPOSITES = (
    "pos_PminusE_land_mean",
    "neg_PminusE_land_mean",
    "pos_PminusE_sea_mean",
    "neg_PminusE_sea_mean",
)

GLOBAL_MEAN_3D_VARS = {
    "specific_humidity": {"var_type": "tendencies", "scale": 1e-7},
    "air_temperature": {"var_type": "tendencies", "scale": 1e-4},
    "liquid_ice_temperature": {"var_type": "tendencies", "scale": 1e-4},
    "cloud_water_ice": {"var_type": "tendencies", "scale": 5e-8},
    "precipitating_water": {"var_type": "tendencies", "scale": 1e-8},
    "total_water": {"var_type": "tendencies", "scale": 1e-7},
    "vertical_wind": {"var_type": "states", "scale": 0.02},
}

GLOBAL_2D_MAPS = {
    "surface_pressure": {"var_type": "tendencies", "scale": 0.1},
    "column_integrated_heating": {"var_type": "tendencies", "scale": 1000},
    "minus_column_integrated_moistening": {"var_type": "tendencies", "scale": 10},
    "vertical_wind_level_40": {"var_type": "states", "scale": 0.05},
    "total_precipitation": {"var_type": "states", "scale": None},
    "latent_heat_flux": {"var_type": "states", "scale": 200},
    "sensible_heat_flux": {"var_type": "states", "scale": 100},
}

MAPPABLE_VAR_KWARGS = {
    "coord_x_center": "x",
    "coord_y_center": "y",
    "coord_x_outer": "x_interface",
    "coord_y_outer": "y_interface",
    "coord_vars": {
        "lonb": ["y_interface", "x_interface", "tile"],
        "latb": ["y_interface", "x_interface", "tile"],
        "lon": ["y", "x", "tile"],
        "lat": ["y", "x", "tile"],
    },
}


__all__ = [
    "SFC_VARIABLES",
    "HI_RES_DIAGS_MAPPING",
    "HI_RES_DIAGS_DIMS",
    "VARS_FROM_ZARR",
    "GRID_VARS",
    "ABS_VARS",
    "LEVEL_VARS",
    "GLOBAL_MEAN_2D_VARS",
    "DIURNAL_VAR_MAPPING",
    "DQ_MAPPING",
    "DQ_PROFILE_MAPPING",
    "PROFILE_COMPOSITES",
    "GLOBAL_MEAN_3D_VARS",
    "GLOBAL_2D_MAPS",
    "MAPPABLE_VAR_KWARGS",
]
