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
EAST_WIND = "eastward_wind_after_physics"
NORTH_WIND = "northward_wind_after_physics"
SST = "ocean_surface_temperature"
TSFC = "surface_temperature"
MASK = "land_sea_mask"

# following variables are required no matter what feature set is being used
TENDENCY_TO_STATE_NAME: Mapping[Hashable, Hashable] = {
    "dQ1": TEMP,
    "dQ2": SPHUM,
    "dQu": EAST_WIND,
    "dQv": NORTH_WIND,
    "dQx_wind": "x_wind",
    "dQy_wind": "y_wind",
    "dQp": DELP,
}
STATE_NAME_TO_TENDENCY = {value: key for key, value in TENDENCY_TO_STATE_NAME.items()}
PREPHYSICS_OVERRIDES = [
    "override_for_time_adjusted_total_sky_downward_shortwave_flux_at_surface",
    "override_for_time_adjusted_total_sky_net_shortwave_flux_at_surface",
    "override_for_time_adjusted_total_sky_downward_longwave_flux_at_surface",
    "ocean_surface_temperature",
]


def is_state_update_variable(key, state: State):
    if key in state.keys() and key not in TENDENCY_TO_STATE_NAME:
        # the second check is to exclude derived variables such as dQu,v
        return True
    elif key == TOTAL_PRECIP_RATE:
        # Special case where models predict precip rate which is
        # converted to state update on accumulated precip
        return True
    else:
        return False
