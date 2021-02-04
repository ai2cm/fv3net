import numpy as np
import pytest
import xarray as xr
from vcm.cubedsphere.constants import COORD_X_CENTER, COORD_Y_CENTER
from vcm.select import (
    mask_to_surface_type,
    get_latlon_grid_coords,
    RegionOfInterest,
    zonal_average_approximate,
)


def test_zonal_average_approximate():
    lat = xr.DataArray(np.linspace(0, 9, 10), dims=["x"]).rename("lat")
    data = xr.DataArray(
        [[i * j for i in range(10)] for j in [0, 1]], dims=["z", "x"]
    ).rename("data")
    zonal_avg = zonal_average_approximate(lat, data, bins=np.arange(0, 10, 2),)
    # bins are (low, high]
    np.testing.assert_allclose(zonal_avg.isel(z=0), np.zeros(4))
    np.testing.assert_allclose(zonal_avg.isel(z=1), [1.5, 3.5, 5.5, 7.5])


@pytest.fixture()
def test_surface_type_grid():
    centered_coords = {COORD_Y_CENTER: [0], COORD_X_CENTER: [0, 1, 2]}
    slmsk_grid = xr.DataArray(
        [[0, 1, 2]], dims=[COORD_Y_CENTER, COORD_X_CENTER], coords=centered_coords
    )
    ds_slmsk = xr.Dataset({"land_sea_mask": slmsk_grid, "checkvar": slmsk_grid})

    return ds_slmsk


@pytest.fixture()
def test_latlon_grid():
    corner_coords = {"tile": [1], COORD_Y_CENTER: range(10), COORD_X_CENTER: range(10)}
    lat = xr.DataArray(
        [[range(10) for i in range(10)]],
        dims=["tile", COORD_Y_CENTER, COORD_X_CENTER],
        coords=corner_coords,
    )
    lon = xr.DataArray(
        [[[i for j in range(10)] for i in range(10)]],
        dims=["tile", COORD_Y_CENTER, COORD_X_CENTER],
        coords=corner_coords,
    )
    grid = xr.Dataset({"lon": lon, "lat": lat})
    return grid


def test_mask_to_surface_type(test_surface_type_grid):
    sea = mask_to_surface_type(test_surface_type_grid, "sea").checkvar.values
    np.testing.assert_equal(sea, np.array([[0, np.nan, np.nan]]))
    land = mask_to_surface_type(test_surface_type_grid, "land").checkvar.values
    np.testing.assert_equal(land, np.array([[np.nan, 1, np.nan]]))
    seaice = mask_to_surface_type(test_surface_type_grid, "seaice").checkvar.values
    np.testing.assert_equal(seaice, np.array([[np.nan, np.nan, 2]]))
    no_mask = mask_to_surface_type(test_surface_type_grid, "global").checkvar.values
    np.testing.assert_equal(no_mask, np.array([[0, 1, 2]]))


def test_get_latlon_grid_coords(test_latlon_grid):
    test_exact_pt = get_latlon_grid_coords(
        test_latlon_grid,
        lat=3,
        lon=2,
        init_search_width=0.5,
        search_width_increment=0.5,
        max_search_width=3,
    )
    assert test_exact_pt[COORD_X_CENTER] == 3
    assert test_exact_pt[COORD_Y_CENTER] == 2
    test_within_radius = get_latlon_grid_coords(
        test_latlon_grid,
        lat=3.25,
        lon=2.25,
        init_search_width=0.45,
        search_width_increment=0.5,
        max_search_width=3,
    )
    assert test_within_radius[COORD_X_CENTER] == 3
    assert test_within_radius[COORD_Y_CENTER] == 2
    test_multiple_pts = get_latlon_grid_coords(
        test_latlon_grid,
        lat=9,
        lon=9,
        init_search_width=2.5,
        search_width_increment=0.5,
        max_search_width=3,
    )
    assert test_multiple_pts[COORD_X_CENTER] in [9, 8, 7]
    assert test_multiple_pts[COORD_Y_CENTER] in [9, 8, 7]
    with pytest.raises(ValueError):
        test_no_match = get_latlon_grid_coords(
            test_latlon_grid,
            lat=13.5,
            lon=12.5,
            init_search_width=0.45,
            search_width_increment=0.5,
            max_search_width=3,
        )
        del test_no_match  # to placate linter


@pytest.mark.parametrize(
    "lat_bounds, lon_bounds, expected_avg",
    [
        pytest.param((1.0, 4.0), (1.0, 4.0), 2.0, id="test area averaging"),
        pytest.param((0.0, 3.0), (0.0, 3.0), 1.0, id="test lat/lon selection"),
    ],
)
def test_roi_average(lat_bounds, lon_bounds, expected_avg):
    roi = RegionOfInterest(lat_bounds, lon_bounds)
    xrange, yrange = 5, 5
    spike_value = 3
    area = [
        [3.0 if x == 3 and y == 3 else 1.0 for x in range(xrange)]
        for y in range(yrange)
    ]
    var = [
        [spike_value if x == 3 and y == 3 else 1.0 for x in range(xrange)]
        for y in range(yrange)
    ]
    ds = xr.Dataset(
        {
            "lat": (
                ["x", "y"],
                np.array([[x for x in range(xrange)] for y in range(yrange)]),
            ),
            "lon": (
                ["x", "y"],
                np.array([[y for x in range(xrange)] for y in range(yrange)]),
            ),
            "var": (["x", "y"], var),
            "area": (["x", "y"], area),
        }
    )
    assert roi.average(ds)["var"].values == expected_avg
