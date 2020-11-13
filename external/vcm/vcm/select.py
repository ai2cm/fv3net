"""
This module is for functions that select subsets of the data
"""
import numpy as np
from typing import Tuple, Hashable, Union, Sequence
import xarray as xr
from dataclasses import dataclass

from vcm.cubedsphere.constants import (
    COORD_X_CENTER,
    COORD_Y_CENTER,
    VAR_LAT_CENTER,
    VAR_LON_CENTER,
)


def zonal_average_approximate(
    lat: xr.DataArray,
    data: Union[xr.DataArray, xr.Dataset],
    bins: Sequence[float] = np.arange(-90, 90, 2),
    lat_dim_name: str = "lat",
):
    data = data.assign_coords(lat=lat)
    grouped = data.groupby_bins("lat", bins=bins)
    output = (
        grouped.mean()
        .drop_vars(lat.name, errors="ignore")
        .rename({lat_dim_name + "_bins": lat_dim_name})
    )
    lats_mid = [lat.item().mid for lat in output[lat_dim_name]]
    return output.assign_coords({lat_dim_name: lats_mid})


def meridional_ring(lon=0, n=180):
    attrs = {"description": f"Lon = {lon}"}
    lat = np.linspace(-90, 90, n)
    lon = np.ones_like(lat) * lon

    return {
        "lat": xr.DataArray(lat, dims="sample", attrs=attrs),
        "lon": xr.DataArray(lon, dims="sample", attrs=attrs),
    }


def zonal_ring(lat=45, n=360):
    attrs = {"description": f"Lat = {lat}"}
    lon = np.linspace(0, 360, n)
    lat = np.ones_like(lon) * lat

    return {
        "lat": xr.DataArray(lat, dims="sample", attrs=attrs),
        "lon": xr.DataArray(lon, dims="sample", attrs=attrs),
    }


@dataclass
class RegionOfInterest:
    lat_bounds: Tuple[float]
    lon_bounds: Tuple[float]

    def average(self, dataset):
        return _roi_average(dataset, self.lat_bounds, self.lon_bounds)


def _roi_average(
    dataset: xr.Dataset,
    lat_bounds: Tuple[float],
    lon_bounds: Tuple[float],
    dims: Tuple[Hashable] = None,
):
    """Average a dataset over a region of interest
    Args:
        dataset: the data to average, must contain, lat, lon, and area variables
        lat_bounds, lon_bounds: the bounds of the regional box
        dims: the spacial dimensions to average over.
    """

    if dims is None:
        dims = dataset["lat"].dims

    stacked = dataset.stack(space=dims)
    grid = stacked
    lon_bounds_pos = [lon % 360.0 for lon in lon_bounds]
    lat_mask = (grid.lat > lat_bounds[0]) & (grid.lat < lat_bounds[1])
    lon_mask = (grid.lon > lon_bounds_pos[0]) & (grid.lon < lon_bounds_pos[1])

    region = stacked.sel(space=lat_mask * lon_mask)
    out = (region * region.area).mean("space") / region.area.mean("space")

    for key in out:
        out[key].attrs.update(dataset[key].attrs)
    out.attrs.update(dataset.attrs)
    return out


def mask_to_surface_type(ds, surface_type, surface_type_var="land_sea_mask"):
    """
    Args:
        ds: xarray dataset, must have variable slmsk
        surface_type: one of ['sea', 'land', 'seaice', 'global']
    Returns:
        input dataset masked to the surface_type specified
    """
    if surface_type == "global":
        return ds
    elif surface_type not in ["sea", "land", "seaice"]:
        raise ValueError(
            "Must mask to surface_type in ['sea', 'land', 'seaice', 'global']."
        )
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
    coord_x_center=COORD_X_CENTER,
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
