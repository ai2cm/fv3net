from typing import List, Mapping, Hashable, Iterable

TEMP = "air_temperature"
TOTAL_WATER = "total_water"
SPHUM = "specific_humidity"
DELP = "pressure_thickness_of_atmospheric_layer"
PRECIP_RATE = "surface_precipitation_rate"
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
}


def filter_matching(variables: Iterable[str], split: str, prefix: str) -> List[str]:
    """Get sequences of tendency and storage variables from diagnostics config."""
    return [
        variable.split(split)[0][len(prefix) :]
        for variable in variables
        if variable.startswith(prefix) and split in variable
    ]


def filter_storage(variables: Iterable[str]) -> List[str]:
    return filter_matching(variables, split="_path_due_to_", prefix="storage_of_")


def filter_tendency(variables: Iterable[str]) -> List[str]:
    return filter_matching(variables, split="_due_to_", prefix="tendency_of_")
