import numpy as np
import xarray as xr
from ..cubedsphere import rename_centered_xy_coords


def convert_vars_to_lat_lon_coords(
        da_x: xr.DataArray,
        da_y: xr.DataArray,
        grid: xr.Dataset
):
    """

    Args:
        da_x: x component of variable (defined on top/bottom cell edges), e.g. u, du_dt
        da_y: y component of variable (defined on left/right cell edges), e.g. v, dv_dt
        grid: dataset of lat/lon at cell centers and edges

    Returns:
        lon_component , lat_component: data arrays of the lon and lat components of the variable
    """
    x_var_centered = rename_centered_xy_coords(0.5 * (da_x + da_x.shift(grid_y=1))[:, 1:, :])
    y_var_centered = rename_centered_xy_coords(0.5 * (da_y + da_y.shift(grid_x=1))[:, :, 1:])

    (e1_lon, e1_lat), (e2_lon, e2_lat) = _get_local_basis_in_spherical_coords(grid)
    lon_unit_vec_cartesian, lat_unit_vec_cartesian = _lon_lat_unit_vectors_to_cartesian(grid)
    e1_cartesian = _spherical_to_cartesian_basis(e1_lon, e1_lat, lon_unit_vec_cartesian, lat_unit_vec_cartesian)
    e2_cartesian = _spherical_to_cartesian_basis(e2_lon, e2_lat, lon_unit_vec_cartesian, lat_unit_vec_cartesian)

    denom = (_dot(e1_cartesian, lon_unit_vec_cartesian) * _dot(e2_cartesian, lat_unit_vec_cartesian) -
             _dot(e2_cartesian, lon_unit_vec_cartesian) * _dot(e1_cartesian, lat_unit_vec_cartesian))
    lon_component = (_dot(e2_cartesian, lat_unit_vec_cartesian) * x_var_centered -
                _dot(e1_cartesian, lat_unit_vec_cartesian) * y_var_centered) / denom
    lat_component = (_dot(e2_cartesian, lon_unit_vec_cartesian) * x_var_centered -
                _dot(e1_cartesian, lon_unit_vec_cartesian) * y_var_centered) / denom

    return lon_component , lat_component


def _deg_to_radians(deg):
    """

    Args:
        deg:

    Returns:

    """
    return deg * np.pi / 180.


def _dot(v1, v2):
    """
    Wrote this because applying np.dot to dataset did not behave as expected / slow
    """
    return sum([v1[i] * v2[i] for i in range(len(v1))])


def _spherical_to_cartesian_basis(
        lon_component,
        lat_component,
        lon_unit_vec,
        lat_unit_vec
):
    """
    Convert a vector from lat/lon basis to cartesian.
    Assumes vector lies on surface of sphere, i.e. spherical basis r component is zero
    Args:
        lon_component:
        lat_component:
        lon_unit_vec:
        lat_unit_vec:

    Returns:

    """
    [x, y, z] = [
        lon_component * lon_unit_vec[i] + lat_component * lat_unit_vec[i]
        for i in range(3)]
    norm = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    x /= norm
    y /= norm
    z /= norm
    return x, y, z


def _lon_lat_unit_vectors_to_cartesian(grid):
    lon_unit_vec = (
        -np.sin(grid.grid_lont),
        np.cos(grid.grid_lont),
        0)
    lat_unit_vec = (
        np.cos(np.pi / 2 - grid.grid_latt) * np.cos(grid.grid_lont),
        np.cos(np.pi / 2 - grid.grid_latt) * np.sin(grid.grid_lont),
        -np.sin(np.pi / 2 - grid.grid_latt))
    return lon_unit_vec, lat_unit_vec


def _lon_diff(corner1, corner2):
    lon_diff = (corner2 - corner1)
    # this handles the prime meridian case
    lon_diff = lon_diff \
        .where(abs(lon_diff) < 180.,
               np.sign(lon_diff) * (abs(lon_diff) - 360))
    return lon_diff


def _get_local_basis_in_spherical_coords(grid):
    """
    Approximates the lon/lat unit vector at cell center as equal to the x/y
    vectors at the bottom left corner written in the lat/lon basis.
    This approximation breaks down near the poles.
    Args:
        grid: xarray dataset with lon/lat defined on corners

    Returns:
        lon_hat & lat_hat: tuples that define unit vectors in lat/lon coordinates
        at the center of each cell.
    """
    xhat_lon_component = 0.5 * _deg_to_radians(
        _lon_diff(grid.grid_lon, grid.grid_lon.shift(grid_x=-1))[:, :-1, :-1]) \
        .rename({'grid_x': 'grid_xt', 'grid_y': 'grid_yt'})
    yhat_lon_component = 0.5 * _deg_to_radians(
        _lon_diff(grid.grid_lon, grid.grid_lon.shift(grid_y=-1))[:, :-1, :-1]) \
        .rename({'grid_x': 'grid_xt', 'grid_y': 'grid_yt'})
    xhat_lat_component = 0.5 * _deg_to_radians(grid.grid_lat.shift(grid_x=-1) - grid.grid_lat)[:, :-1, :-1] \
        .rename({'grid_x': 'grid_xt', 'grid_y': 'grid_yt'})
    yhat_lat_component = 0.5 * _deg_to_radians(grid.grid_lat.shift(grid_y=-1) - grid.grid_lat)[:, :-1, :-1] \
        .rename({'grid_x': 'grid_xt', 'grid_y': 'grid_yt'})
    return (xhat_lon_component, xhat_lat_component), (yhat_lon_component, yhat_lat_component)



