import numpy as np
import xarray as xr


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
        _lon_diff(grid.grid_lon, grid.grid_lon.shift(grid_x=-1))[:, :-1, :-1]
    ).rename({"grid_x": "grid_xt", "grid_y": "grid_yt"})
    yhat_lon_component = np.deg2rad(
        _lon_diff(grid.grid_lon, grid.grid_lon.shift(grid_y=-1))[:, :-1, :-1]
    ).rename({"grid_x": "grid_xt", "grid_y": "grid_yt"})
    xhat_lat_component = np.deg2rad(grid.grid_lat.shift(grid_x=-1) - grid.grid_lat)[
        :, :-1, :-1
    ].rename({"grid_x": "grid_xt", "grid_y": "grid_yt"})
    yhat_lat_component = np.deg2rad(grid.grid_lat.shift(grid_y=-1) - grid.grid_lat)[
        :, :-1, :-1
    ].rename({"grid_x": "grid_xt", "grid_y": "grid_yt"})
    return (
        (xhat_lon_component, xhat_lat_component),
        (yhat_lon_component, yhat_lat_component),
    )


def mask_antimeridian_quads(lonb: np.ndarray, central_longitude: float):

    """ Computes mask of grid quadrilaterals bisected by a projection system's antimeridian,
    in order to avoid cartopy plotting artifacts
    
    Args:
    
        lonb (np.ndarray): 
            Array of grid edge longitudes, of dimensions (npy + 1, npx + 1, tile) 
        central_longitude (float): 
            Central longitude from which the antimeridian is computed
    
    Returns:
    
        mask (np.ndarray): 
            Boolean array of grid centers, False = excluded, of dimensions (npy, npx, tile) 

    
    Example:
    
        masked_array = np.where(
            mask_antimeridian_quads(lonb, central_longitude),
            array,
            np.nan
        )
    """

    antimeridian = (central_longitude + 180.0) % 360.0
    mask = np.full([lonb.shape[0] - 1, lonb.shape[1] - 1, lonb.shape[2]], True)
    for tile in range(6):
        tile_lonb = lonb[:, :, tile]
        tile_mask = mask[:, :, tile]
        for ix in range(tile_lonb.shape[0] - 1):
            for iy in range(tile_lonb.shape[1] - 1):
                vertex_indices = ([ix, ix + 1, ix, ix + 1], [iy, iy, iy + 1, iy + 1])
                vertices = tile_lonb[vertex_indices]
                if (
                    sum(_periodic_equal_or_less_than(vertices, antimeridian)) != 4
                    and sum(_periodic_greater_than(vertices, antimeridian)) != 4
                    and sum((_periodic_difference(vertices, antimeridian) < 90.0)) == 4
                ):
                    tile_mask[ix, iy] = False
        mask[:, :, tile] = tile_mask

    return mask


def _periodic_equal_or_less_than(x1, x2, period=360.0):

    """ Compute whether x1 is less than or equal to x2, where 
    the difference between the two is the shortest distance on a periodic domain 
    
    Args:
    
        x1 (float), x2 (float):
            Values to be compared
        Period (float, optional): 
            Period of domain. Default 360 (degrees). 
    
    Returns:
    
        Less_than_or_equal (Bool): 
            Whether x1 is less than or equal to x2
    
    
    """

    return np.where(
        np.abs(x1 - x2) <= period / 2.0,
        np.where(x1 - x2 <= 0, True, False),
        np.where(
            x1 - x2 >= 0,
            np.where(x1 - (x2 + period) <= 0, True, False),
            np.where((x1 + period) - x2 <= 0, True, False),
        ),
    )


def _periodic_greater_than(x1, x2, period=360.0):

    """ Compute whether x1 is greater than x2, where 
    the difference between the two is the shortest distance on a periodic domain 
    
    Args:
    
        x1 (float), x2 (float):
            Values to be compared
        Period (float, optional): 
            Period of domain. Default 360 (degrees). 
    
    Returns:
    
        Greater_than (Bool): 
            Whether x1 is greater than x2
    
    
    """

    return np.where(
        np.abs(x1 - x2) <= period / 2.0,
        np.where(x1 - x2 > 0, True, False),
        np.where(
            x1 - x2 >= 0,
            np.where(x1 - (x2 + period) > 0, True, False),
            np.where((x1 + period) - x2 > 0, True, False),
        ),
    )


def _periodic_difference(x1, x2, period=360.0):

    """ Compute difference between x1 and x2, where 
    the difference is the shortest distance on a periodic domain 
    
    Args:
    
        x1 (float), x2 (float):
            Values to be compared
        Period (float, optional): 
            Period of domain. Default 360 (degrees). 
    
    Returns:
    
        Difference (float): 
            Difference between x1 and x2
    
    
    """

    return np.where(
        np.abs(x1 - x2) <= period / 2.0,
        x1 - x2,
        np.where(x1 - x2 >= 0, x1 - (x2 + period), (x1 + period) - x2),
    )
