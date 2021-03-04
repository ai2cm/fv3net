from typing import Mapping, Hashable

TEMP = "air_temperature"
TOTAL_WATER = "total_water"
SPHUM = "specific_humidity"
DELP = "pressure_thickness_of_atmospheric_layer"
PRECIP_RATE = "surface_precipitation_rate"
TOTAL_PRECIP = "total_precipitation"  # has units of m
AREA = "area_of_grid_cell"
EAST_WIND_AFTER_PHYSICS = "eastward_wind_after_physics"
NORTH_WIND_AFTER_PHYSICS = "northward_wind_after_physics"
EAST_WIND = "eastward_wind"
NORTH_WIND = "northward_wind"
CLOUD_ICE = "cloud_ice_mixing_ratio"
CLOUD_WATER = "cloud_water_mixing_ratio"
GRAUPEL = "graupel_mixing_ratio"
RAIN = "rain_mixing_ratio"
SNOW = "snow_mixing_ratio"
OZONE = "ozone_mixing_ratio"

# following variables are required no matter what feature set is being used
TENDENCY_TO_STATE_NAME: Mapping[Hashable, Hashable] = {
    "dQ1": TEMP,
    "dQ2": SPHUM,
    "dQu": EAST_WIND_AFTER_PHYSICS,
    "dQv": NORTH_WIND_AFTER_PHYSICS,
    "tendency_of_air_temperature_due_to_fv3_physics": TEMP,
    "tendency_of_specific_humidity_due_to_fv3_physics": SPHUM,
    "tendency_of_eastward_wind_due_to_fv3_physics": EAST_WIND,
    "tendency_of_northward_wind_due_to_fv3_physics": NORTH_WIND,
    "tendency_of_cloud_ice_mixing_ratio_due_to_fv3_physics": CLOUD_ICE,
    "tendency_of_cloud_water_mixing_ratio_due_to_fv3_physics": CLOUD_WATER,
    "tendency_of_graupel_mixing_ratio_due_to_fv3_physics": GRAUPEL,
    "tendency_of_ozone_mixing_ratio_due_to_fv3_physics": OZONE,
    "tendency_of_rain_mixing_ratio_due_to_fv3_physics": RAIN,
    "tendency_of_snow_mixing_ratio_due_to_fv3_physics": SNOW,
}
