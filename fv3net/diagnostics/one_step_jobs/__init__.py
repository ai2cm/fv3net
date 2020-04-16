INIT_TIME_DIM = "initial_time"
FORECAST_TIME_DIM = "forecast_time"
DELTA_DIM = "model_run"
VAR_TYPE_DIM = "var_type"
STEP_DIM = "step"
OUTPUT_NC_FILENAME = "one_step_diag_data.nc"
ONE_STEP_ZARR = "big.zarr"
SFC_VARIABLES = (
    "DSWRFtoa",
    "DSWRFsfc",
    "USWRFtoa",
    "USWRFsfc",
    "DLWRFsfc",
    "ULWRFtoa",
    "ULWRFsfc",
)
GRID_VARS = ("lat", "lon", "latb", "lonb", "area", "land_sea_mask")
VARS_FROM_ZARR = (
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
) + SFC_VARIABLES
ABS_VARS = ["psurf", "precipitable_water", "total_heat"]
GLOBAL_MEAN_2D_VARS = {
    "psurf_abs": {VAR_TYPE_DIM: ["tendencies"], "scale": [0.12]},
    "precipitable_water_abs": {VAR_TYPE_DIM: ["tendencies"], "scale": [0.0012]},
    "total_heat": {VAR_TYPE_DIM: ["tendencies", "states"], "scale": [1000, None]},
    "latent_heat_flux": {VAR_TYPE_DIM: ["states"], "scale": [None]},
    "sensible_heat_flux": {VAR_TYPE_DIM: ["states"], "scale": [None]},
    "total_precipitation": {VAR_TYPE_DIM: ["states"], "scale": [None]},
}

DIURNAL_VAR_MAPPING = {
    "net_heating_diurnal": {
        "coarse": {"name": "column_integrated_heating", VAR_TYPE_DIM: "tendencies"},
        "hi-res": {"name": "net_heating_physics", VAR_TYPE_DIM: "states"},
        "scale": 500,
    },
    "net_precipitation_diurnal": {
        "coarse": {"name": "column_integrated_moistening", VAR_TYPE_DIM: "tendencies"},
        "hi-res": {"name": "net_precipitation_physics", VAR_TYPE_DIM: "states"},
        "scale": 10,
    },
    "vertical_wind_diurnal": {
        "coarse": {"name": "vertical_wind_level_40", VAR_TYPE_DIM: "states"},
        "hi-res": {"name": "vertical_wind_level_40", VAR_TYPE_DIM: "states"},
        "scale": 0.05,
    },
}

DQ_MAPPING = {
    "Q1": {
        "hi-res_name": "net_heating",
        "coarse_name": "column_integrated_heating",
        "scale": 1000,
    },
    "Q2": {
        "hi-res_name": "net_precipitation",
        "coarse_name": "column_integrated_moistening",
        "scale": 50,
    },
}

DQ_PROFILE_MAPPING = {
    "air_temperature": {"name": "dQ1", VAR_TYPE_DIM: "tendencies", "scale": 1.5e-4},
    "specific_humidity": {"name": "dQ2", VAR_TYPE_DIM: "tendencies", "scale": 2e-7},
    "vertical_wind": {"name": "dW", VAR_TYPE_DIM: "states", "scale": 0.05},
}

GLOBAL_MEAN_3D_VARS = {
    "specific_humidity": {VAR_TYPE_DIM: "tendencies", "scale": 1e-7},
    "air_temperature": {VAR_TYPE_DIM: "tendencies", "scale": 1e-4},
    "cloud_water_ice": {VAR_TYPE_DIM: "tendencies", "scale": 5e-8},
    "precipitating_water": {VAR_TYPE_DIM: "tendencies", "scale": 1e-8},
    "vertical_wind": {VAR_TYPE_DIM: "states", "scale": 0.05},
}

GLOBAL_2D_MAPS = {
    "psurf": {VAR_TYPE_DIM: "tendencies", "scale": None},
    "column_integrated_heating": {VAR_TYPE_DIM: "tendencies", "scale": 1000},
    "column_integrated_moistening": {VAR_TYPE_DIM: "tendencies", "scale": 50},
    "vertical_wind_level_40": {VAR_TYPE_DIM: "states", "scale": 0.05},
    "total_precipitation": {VAR_TYPE_DIM: "states", "scale": None},
    "latent_heat_flux": {VAR_TYPE_DIM: "states", "scale": 100},
    "sensible_heat_flux": {VAR_TYPE_DIM: "states", "scale": 100},
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
