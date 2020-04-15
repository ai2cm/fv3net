INIT_TIME_DIM = 'initial_time'
FORECAST_TIME_DIM = 'forecast_time'
DELTA_DIM = 'model_run'
VAR_TYPE_DIM = 'var_type'
STEP_DIM = 'step'
OUTPUT_NC_FILENAME = 'one_step_diag_data.nc'
ONE_STEP_ZARR = 'big.zarr'
SFC_VARIABLES = (
    "DSWRFtoa",
    "DSWRFsfc",
    "USWRFtoa",
    "USWRFsfc",
    "DLWRFsfc",
    "ULWRFtoa",
    "ULWRFsfc"
)
GRID_VARS = ('lat', 'lon', 'latb', 'lonb', 'area', 'land_sea_mask')
VARS_FROM_ZARR = (
    'specific_humidity',
    'cloud_ice_mixing_ratio',
    'cloud_water_mixing_ratio',
    'rain_mixing_ratio',
    'snow_mixing_ratio',
    'graupel_mixing_ratio',
    'vertical_wind',
    'air_temperature',
    'pressure_thickness_of_atmospheric_layer',
    "latent_heat_flux",
    "sensible_heat_flux",
    "total_precipitation"
) + SFC_VARIABLES
ABS_VARS = ['psurf', 'precipitable_water', 'total_heat']
GLOBAL_MEAN_2D_VARS = {
    'psurf_abs': {
        VAR_TYPE_DIM: 'tendency',
        "scale": 0.12
    },
    'precipitable_water_abs': {
        VAR_TYPE_DIM: 'tendency',
        "scale": 0.00012
    },
    'total_heat_abs': {},
    'precipitable_water': {},
#     'cloud_water_ice': None,
    'total_heat': {}
}
GLOBAL_MEAN_3D_VARS = ["specific_humidity", "air_temperature", "vertical_wind"]
DIURNAL_VAR_MAPPING = {
    "net_heating_diurnal": {
        "coarse": {
            "name": "column_integrated_heating",
            VAR_TYPE_DIM: 'tendencies'
        },
        "hi-res": {
            "name": "net_heating_physics",
            VAR_TYPE_DIM: "states"
        }
    },
    "net_precipitation_diurnal": {
        "coarse": {
            "name": "column_integrated_moistening",
            VAR_TYPE_DIM: 'tendencies'
        },
        "hi-res": {
            "name": "net_precipitation_physics",
            VAR_TYPE_DIM: "states"
        }
    },
    "vertical_wind_diurnal": {
        "coarse": {
            "name": "vertical_wind_level_40",
            VAR_TYPE_DIM: 'states'
        },
        "hi-res": {
            "name": "vertical_wind_level_40",
            VAR_TYPE_DIM: "states"
        }
    },
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
        "lat": ["y", "x", "tile"] 
    }
}