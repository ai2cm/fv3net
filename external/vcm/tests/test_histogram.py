import numpy as np
import xarray as xr
from vcm.calc.histogram import histogram, histogram2d


def test_histogram():
    data = xr.DataArray(
        np.reshape(np.arange(0, 40, 2), (5, 4)), dims=["x", "y"], name="temperature"
    )
    coords = {"temperature_bins": [0, 30]}
    expected_count = xr.DataArray([15, 5], coords=coords, dims="temperature_bins")
    expected_width = xr.DataArray([30, 10], coords=coords, dims="temperature_bins")
    count, width = histogram(data, bins=[0, 30, 40])
    xr.testing.assert_equal(count, expected_count)
    xr.testing.assert_equal(width, expected_width)


def test_histogram2d():
    data_x = np.arange(0, 40, 2)
    data_y = np.arange(0, 20, 1)
    bins = [np.array([0, 20, 40]), np.array([0, 10, 20])]
    var1 = xr.DataArray(np.reshape(data_x, (5, 4)), dims=["x", "y"], name="temp")
    var2 = xr.DataArray(np.reshape(data_y, (5, 4)), dims=["x", "y"], name="humidity")
    var2 = var2.transpose("y", "x")  # vcm.histogram2d should handle this
    temp_coords = {"temp_bins": [0, 20]}
    sphum_coords = {"humidity_bins": [0, 10]}
    expected_count = xr.DataArray(
        np.histogram2d(data_x, data_y, bins)[0],
        coords={**temp_coords, **sphum_coords},
        dims=["temp_bins", "humidity_bins"],
    )
    expected_xwidth = xr.DataArray([20, 20], coords=temp_coords, dims="temp_bins")
    expected_ywidth = xr.DataArray([10, 10], coords=sphum_coords, dims="humidity_bins")
    count, xwidth, ywidth = histogram2d(var1, var2, bins=bins)
    xr.testing.assert_equal(count, expected_count)
    xr.testing.assert_equal(xwidth, expected_xwidth)
    xr.testing.assert_equal(ywidth, expected_ywidth)
