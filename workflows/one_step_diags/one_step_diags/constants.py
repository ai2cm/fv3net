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
REPORT_TITLE = "One-step diagnostics"
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
