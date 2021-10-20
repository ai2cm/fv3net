import numpy as np
import xarray as xr
from vcm.calc.histogram import histogram


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
