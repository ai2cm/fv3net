import numpy as np
import xarray as xr
from vcm.cubedsphere.coarsen import shift_edge_var_to_center
from vcm.cubedsphere.constants import (
    COORD_X_CENTER,
    COORD_Y_CENTER,
    COORD_X_OUTER,
    COORD_Y_OUTER,
    VAR_LON_CENTER,
    VAR_LAT_CENTER,
    VAR_LON_OUTER,
    VAR_LAT_OUTER,
)


def rotate_winds_to_lat_lon_coords(
    da_x: xr.DataArray, da_y: xr.DataArray, grid: xr.Dataset
):
    """ Transforms a vector in the x/y plane into lat/lon coordinates.

    The input x and y components are assumed to be defined at the cell center.
    If they are originally defined on edges, e.g. u and v, they can be centered with
    func shift_edge_var_to_center from vcm.calc.cubedsphere
    Args:
        da_x: x component of variable, e.g. u, du_dt
        da_y: y component of variable, e.g. v, dv_dt
        grid: dataset of lat/lon at cell centers and edges

    Returns:
        lon_component, lat_component: data arrays of the lon and lat components of the
                                      variable
    """
    (e1_lon, e1_lat), (e2_lon, e2_lat) = _get_local_basis_in_spherical_coords(grid)
    lon_unit_vec_cartesian, lat_unit_vec_cartesian = _lon_lat_unit_vectors_to_cartesian(
        grid
    )
    e1_cartesian = _spherical_to_cartesian_basis(
        e1_lon, e1_lat, lon_unit_vec_cartesian, lat_unit_vec_cartesian
    )
    e2_cartesian = _spherical_to_cartesian_basis(
        e2_lon, e2_lat, lon_unit_vec_cartesian, lat_unit_vec_cartesian
    )

    denom = _dot(e1_cartesian, lon_unit_vec_cartesian) * _dot(
        e2_cartesian, lat_unit_vec_cartesian
    ) - _dot(e2_cartesian, lon_unit_vec_cartesian) * _dot(
        e1_cartesian, lat_unit_vec_cartesian
    )
    lon_component = (
        _dot(e2_cartesian, lat_unit_vec_cartesian) * da_x
        - _dot(e1_cartesian, lat_unit_vec_cartesian) * da_y
    ) / denom
    lat_component = (
        _dot(e2_cartesian, lon_unit_vec_cartesian) * da_x
        - _dot(e1_cartesian, lon_unit_vec_cartesian) * da_y
    ) / denom

    return lon_component, lat_component


def _dot(v1, v2):
    """ Wrote this because applying np.dot to dataset did not behave as expected / slow
    """
    return sum([v1[i] * v2[i] for i in range(len(v1))])


def _spherical_to_cartesian_basis(
    lon_component, lat_component, lon_unit_vec, lat_unit_vec
):
    """ Convert a vector from lat/lon basis to cartesian.

    Assumes vector lies on surface of sphere, i.e. spherical basis r component is zero
    Args:
        lon_component: coefficient that multiplies lon unit vec at a point on sphere
        lat_component: " " " lat unit vec
        lon_unit_vec: lon unit vec in cartesian basis at point on sphere
        lat_unit_vec: lat unit vec in cartesian basis at point on sphere

    Returns:
        components of the vector in cartesian basis
    """
    [x, y, z] = [
        lon_component * lon_unit_vec[i] + lat_component * lat_unit_vec[i]
        for i in range(3)
    ]
    norm = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    x /= norm
    y /= norm
    z /= norm
    return x, y, z


def _lon_lat_unit_vectors_to_cartesian(grid):
    """

    Args:
        grid: Dataset with lat/lon coordinates defined at edges and cell centers

    Returns:
        lon and lat vectors at the center of each cell, expressed in cartesian basis
    """
    lon_vec_cartesian = (
        -np.sin(np.deg2rad(grid.grid_lont)),
        np.cos(np.deg2rad(grid.grid_lont)),
        0,
    )
    lat_vec_cartesian = (
        np.cos(np.pi / 2 - np.deg2rad(grid.grid_latt))
        * np.cos(np.deg2rad(grid.grid_lont)),
        np.cos(np.pi / 2 - np.deg2rad(grid.grid_latt))
        * np.sin(np.deg2rad(grid.grid_lont)),
        -np.sin(np.pi / 2 - np.deg2rad(grid.grid_latt)),
    )
    return lon_vec_cartesian, lat_vec_cartesian


def _lon_diff(edge1, edge2):
    """ Handles the case where edges are on either side of prime meridian.
    Input units are degrees.

    Args:
        edge1: dataArray of grid_lon (lon at edges)
        edge2: dataArray of grid_lon + n cells offset

    Returns: dataArray of lon difference

    """
    lon_diff = edge2 - edge1
    # this handles the prime meridian case
    lon_diff = lon_diff.where(
        abs(lon_diff) < 180.0, np.sign(lon_diff) * (abs(lon_diff) - 360)
    )
    return lon_diff


def _get_local_basis_in_spherical_coords(grid):
    """ Approximates the lon/lat unit vector at cell center as equal to the x/y
    vectors at the bottom left corner written in the lat/lon basis.
    This approximation breaks down near the poles.

    Args:
        grid: xarray dataset with lon/lat defined on corners

    Returns:
        lon_hat & lat_hat: tuples that define unit vectors in lat/lon coordinates
        at the center of each cell.
    """
    xhat_lon_component = np.deg2rad(
        _lon_diff(grid.grid_lon, grid.grid_lon.shift({COORD_X_OUTER: -1}))[:, :-1, :-1]
    ).rename({COORD_X_OUTER: COORD_X_CENTER, COORD_Y_OUTER: COORD_Y_CENTER})
    yhat_lon_component = np.deg2rad(
        _lon_diff(grid.grid_lon, grid.grid_lon.shift({COORD_Y_OUTER: -1}))[:, :-1, :-1]
    ).rename({COORD_X_OUTER: COORD_X_CENTER, COORD_Y_OUTER: COORD_Y_CENTER})
    xhat_lat_component = np.deg2rad(
        grid.grid_lat.shift({COORD_X_OUTER: -1}) - grid.grid_lat
    )[:, :-1, :-1].rename(
        {COORD_X_OUTER: COORD_X_CENTER, COORD_Y_OUTER: COORD_Y_CENTER}
    )
    yhat_lat_component = np.deg2rad(
        grid.grid_lat.shift({COORD_Y_OUTER: -1}) - grid.grid_lat
    )[:, :-1, :-1].rename(
        {COORD_X_OUTER: COORD_X_CENTER, COORD_Y_OUTER: COORD_Y_CENTER}
    )
    return (
        (xhat_lon_component, xhat_lat_component),
        (yhat_lon_component, yhat_lat_component),
    )


def get_rotated_centered_winds_from_restarts(ds: xr.Dataset):

    """ Get rotated and centered winds from restart wind variables

    Args:

        ds (xr.Dataset):
            Dataset containing 'u' and 'v' restart wind variables on
            staggered, tiled grid; also containing grid variables for centers
            and edges

    Returns:

        u_r, v_r (xr.DataArrays)
            DataArrays of rotated, centered winds


    """

    u_c = shift_edge_var_to_center(ds["u"].drop(labels=COORD_X_CENTER))
    v_c = shift_edge_var_to_center(ds["v"].drop(labels=COORD_Y_CENTER))
    return rotate_winds_to_lat_lon_coords(
        u_c, v_c, ds[[VAR_LON_CENTER, VAR_LAT_CENTER, VAR_LON_OUTER, VAR_LAT_OUTER]]
    )
