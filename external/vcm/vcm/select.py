"""
This module is for functions that select subsets of the data
"""
import numpy as np

from vcm.cubedsphere.constants import (
    COORD_X_CENTER,
    COORD_Y_CENTER,
    VAR_LAT_CENTER,
    VAR_LON_CENTER,
)


def mask_to_surface_type(ds, surface_type, surface_type_var="land_sea_mask"):
    """
    Args:
        ds: xarray dataset, must have variable slmsk
        surface_type: one of ['sea', 'land', 'seaice']
    Returns:
        input dataset masked to the surface_type specified
    """
    if surface_type in ["none", "None", None, "global"]:
        return ds
    elif surface_type not in ["sea", "land", "seaice"]:
        raise ValueError("Must mask to surface_type in ['sea', 'land', 'seaice'].")
    surface_type_codes = {"sea": 0, "land": 1, "seaice": 2}
    mask = ds[surface_type_var].astype(int) == surface_type_codes[surface_type]
    ds_masked = ds.where(mask)
    return ds_masked


def drop_nondim_coords(ds):
    for coord in ds.coords:
        if not isinstance(ds[coord].values, np.ndarray):
            ds = ds.squeeze().drop(coord)
    return ds


def get_latlon_grid_coords(
    grid,
    lat,
    lon,
    var_lat=VAR_LAT_CENTER,
    var_lon=VAR_LON_CENTER,
    coord_x_center=COORD_Y_CENTER,
    coord_y_center=COORD_Y_CENTER,
    init_search_width=0.5,
    search_width_increment=0.5,
    max_search_width=3.0,
):
    """ Convenience function to look up the APPROXIMATE grid coordinates for a
    lat/lon coordinate.

    Args:
        grid: xr dataset with grid coordinate variables
        lat: latitude [deg]
        lon: longitude [deg] convention is 0-360. This will probably not work well for
        longitudes close to the intl date line.
        init_search_width: initial search radius to filter
        search_width_increment: increment if mask does not return any points
        max_search_width: maximum radius to search before giving up

    Returns:
        dict of values that can be passed to xarray .sel()
    """
    search_width = init_search_width
    while search_width <= max_search_width:
        lat_mask = (grid[var_lat] > lat - 1) & (grid[var_lat] < lat + 1)
        lon_mask = (grid[var_lon] > lon - 1) & (grid[var_lon] < lon + 1)
        local_pt = (
            grid[[var_lat, var_lon]]
            .where(lat_mask)
            .where(lon_mask)
            .stack(sample=[coord_x_center, coord_y_center, "tile"])
            .dropna("sample")
        )
        if len(local_pt.sample.values) > 0:
            return {
                "tile": local_pt["tile"].values[0],
                coord_x_center: local_pt[coord_x_center].values[0],
                coord_y_center: local_pt[coord_y_center].values[0],
            }
        else:
            search_width += search_width_increment
    raise ValueError(
        f"No grid points with lat/lon within +/- {max_search_width} deg of {lat, lon}."
    )
