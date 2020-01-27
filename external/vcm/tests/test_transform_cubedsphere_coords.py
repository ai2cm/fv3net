import numpy as np
import pytest
import xarray as xr

from vcm.cubedsphere.transform_cubesphere_coords import (
    _get_local_basis_in_spherical_coords,
    _lon_diff,
    _lon_lat_unit_vectors_to_cartesian,
    rotate_winds_to_lat_lon_coords,
)
from vcm.cubedsphere.constants import (
    COORD_X_CENTER,
    COORD_Y_CENTER,
    COORD_X_OUTER,
    COORD_Y_OUTER,
)


RAD_PER_DEG = np.pi / 180.0


@pytest.fixture()
def test_unit_grid():
    centered_coords = {"tile": [1], COORD_Y_CENTER: [1], COORD_X_CENTER: [1]}
    lon = xr.DataArray(
        [[[1.0]]], dims=["tile", COORD_Y_CENTER, COORD_X_CENTER], coords=centered_coords
    )
    lat = xr.DataArray(
        [[[0.0]]], dims=["tile", COORD_Y_CENTER, COORD_X_CENTER], coords=centered_coords
    )
    corner_coords = {"tile": [1], COORD_Y_OUTER: [1, 2], COORD_X_OUTER: [1, 2]}
    lonb = xr.DataArray(
        [[[359.0, 3.0], [359.0, 3.0]]],
        dims=["tile", COORD_Y_OUTER, COORD_X_OUTER],
        coords=corner_coords,
    )
    latb = xr.DataArray(
        [[[-2.0, -2.0], [2.0, 2.0]]],
        dims=["tile", COORD_Y_OUTER, COORD_X_OUTER],
        coords=corner_coords,
    )
    grid = xr.Dataset({"lon": lon, "lat": lat, "lonb": lonb, "latb": latb})
    return grid


def test_lon_diff(test_unit_grid):
    # test over prime meridian
    lon_diff = _lon_diff(
        test_unit_grid.lonb, test_unit_grid.lonb.shift({COORD_X_OUTER: -1})
    )[:, :-1, :-1]
    assert lon_diff == 4.0


def test_get_local_basis_in_spherical_coords(test_unit_grid):
    xhat, yhat = _get_local_basis_in_spherical_coords(test_unit_grid)
    assert xhat[0].values.flatten()[0] == 4 * RAD_PER_DEG
    assert xhat[1].values.flatten()[0] == 0
    assert yhat[0].values.flatten()[0] == 0
    assert yhat[1].values.flatten()[0] == 4 * RAD_PER_DEG


def test_lon_lat_unit_vectors_to_cartesian(test_unit_grid):
    lonhat, lathat = _lon_lat_unit_vectors_to_cartesian(test_unit_grid)
    assert lonhat == pytest.approx(
        (-np.sin(1.0 * RAD_PER_DEG), np.cos(1.0 * RAD_PER_DEG), 0.0)
    )
    assert lathat == pytest.approx((0.0, 0.0, -1.0))


def test_rotate_winds_to_lat_lon_coords(test_unit_grid):
    x_momentum = xr.DataArray(
        [[[1]]],
        dims=["tile", COORD_Y_CENTER, COORD_X_CENTER],
        coords={"tile": [1], COORD_Y_CENTER: [1], COORD_X_CENTER: [1]},
    )
    y_momentum = xr.DataArray(
        [[[-1]]],
        dims=["tile", COORD_Y_CENTER, COORD_X_CENTER],
        coords={"tile": [1], COORD_Y_CENTER: [1], COORD_X_CENTER: [1]},
    )
    lon_momentum, lat_momentum = rotate_winds_to_lat_lon_coords(
        x_momentum, y_momentum, test_unit_grid
    )
    assert (lon_momentum, lat_momentum) == pytest.approx((1, 1))
