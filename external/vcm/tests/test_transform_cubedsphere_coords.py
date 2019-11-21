import pytest
import numpy as np
import xarray as xr

from vcm.calc.transform_cubesphere_coords import (
    _lon_diff,
    _get_local_basis_in_spherical_coords,
    _lon_lat_unit_vectors_to_cartesian
)

RAD_PER_DEG = np.pi / 180.


@pytest.fixture()
def test_unit_grid():
    centered_coords = {"tile": [1], "grid_yt": [1], "grid_xt": [1]}
    lont_da = xr.DataArray(
        [[[1.]]],
        dims=["tile", "grid_yt", "grid_xt"],
        coords=centered_coords)
    latt_da = xr.DataArray(
        [[[0.]]],
        dims=["tile", "grid_yt", "grid_xt"],
        coords=centered_coords)
    corner_coords = {"tile": [1], "grid_y": [1, 2], "grid_x": [1, 2]}
    lon_grid = xr.DataArray(
        [[[359., 3.], [359., 3.]]],
        dims=["tile", "grid_y", "grid_x"],
        coords=corner_coords
    )
    lat_grid = xr.DataArray(
        [[[-2., -2.], [2., 2.]]],
        dims=["tile", "grid_y", "grid_x"],
        coords=corner_coords
    )
    grid = xr.Dataset({
        "grid_lont": lont_da,
        "grid_latt": latt_da,
        "grid_lon": lon_grid,
        "grid_lat": lat_grid
    })
    return grid


def test_lon_diff(test_unit_grid):
    # test over prime meridian
    lon_diff = _lon_diff(
        test_unit_grid.grid_lon, test_unit_grid.grid_lon.shift(grid_x=-1))[:, :-1, :-1]
    assert lon_diff == 4.


def test_get_local_basis_in_spherical_coords(test_unit_grid):
    xhat, yhat = _get_local_basis_in_spherical_coords(test_unit_grid)
    assert xhat == (4 * RAD_PER_DEG, 0)
    assert yhat == (0, 4 * RAD_PER_DEG)


def test_lon_lat_unit_vectors_to_cartesian(test_unit_grid):
    lonhat, lathat = _lon_lat_unit_vectors_to_cartesian(test_unit_grid)
    assert lonhat == pytest.approx(
        (-np.sin(1. * RAD_PER_DEG),
         np.cos(1. * RAD_PER_DEG),
         0.))
    assert lathat == pytest.approx((0., 0., -1.))
