from typing import Mapping, Hashable
from .types import State


TEMP = "air_temperature"
TOTAL_WATER = "total_water"
CLOUD = "cloud_water_mixing_ratio"
SPHUM = "specific_humidity"
DELP = "pressure_thickness_of_atmospheric_layer"
# [kg/m2/s], due to physics parmameterization
PHYSICS_PRECIP_RATE = "surface_precipitation_rate"
# [kg/m2/s], might also include nudging or ML contributions on top of physics
TOTAL_PRECIP_RATE = "total_precipitation_rate"
TOTAL_PRECIP = "total_precipitation"  # has units of m
AREA = "area_of_grid_cell"
EASTWARD_WIND_AFTER_PHYSICS = "eastward_wind_after_physics"
EASTWARD_WIND = "eastward_wind"
NORTHWARD_WIND = "northward_wind"
SST = "ocean_surface_temperature"
TSFC = "surface_temperature"
MASK = "land_sea_mask"
TIME_KEYS = ["time", "initialization_time"]
X_WIND = "x_wind"
Y_WIND = "y_wind"
EASTWARD_WIND_TENDENCY = "dQu"
NORTHWARD_WIND_TENDENCY = "dQv"
X_WIND_TENDENCY = "dQx_wind"
Y_WIND_TENDENCY = "dQy_wind"

# following variables are required no matter what feature set is being used
TENDENCY_TO_STATE_NAME: Mapping[Hashable, Hashable] = {
    "dQ1": TEMP,
    "dQ2": SPHUM,
    EASTWARD_WIND_TENDENCY: EASTWARD_WIND,
    NORTHWARD_WIND_TENDENCY: NORTHWARD_WIND,
    X_WIND_TENDENCY: X_WIND,
    Y_WIND_TENDENCY: Y_WIND,
    "dQp": DELP,
}
STATE_NAME_TO_TENDENCY = {value: key for key, value in TENDENCY_TO_STATE_NAME.items()}
SURFACE_FLUX_OVERRIDES = [
    "override_for_time_adjusted_total_sky_downward_shortwave_flux_at_surface",
    "override_for_time_adjusted_total_sky_net_shortwave_flux_at_surface",
    "override_for_time_adjusted_total_sky_downward_longwave_flux_at_surface",
]
PREPHYSICS_OVERRIDES = [
    *SURFACE_FLUX_OVERRIDES,
    "ocean_surface_temperature",
    "surface_temperature",
]
A_GRID_WIND_TENDENCIES = {EASTWARD_WIND_TENDENCY, NORTHWARD_WIND_TENDENCY}
D_GRID_WIND_TENDENCIES = {X_WIND_TENDENCY, Y_WIND_TENDENCY}
TENDENCY_NAMES = set(TENDENCY_TO_STATE_NAME) | A_GRID_WIND_TENDENCIES

FV3GFS = "fv3gfs"


def is_state_update_variable(key, state: State):
    if key in state.keys() and key not in TENDENCY_NAMES:
        return True
    elif key == TOTAL_PRECIP_RATE:
        # Special case where models predict precip rate which is
        # converted to state update on accumulated precip
        return True
    else:
        return False


def is_tendency_variable(key):
    return key in TENDENCY_NAMES
