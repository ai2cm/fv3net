import numpy as np
from datetime import datetime
import pytest
import xarray as xr

from diagnostics_utils._diurnal_cycle import _local_time, bin_diurnal_cycle


@pytest.mark.parametrize(
    "time_gmt, expected",
    (
        [datetime(2016, 1, 1, 0, 0, 0), [0, 6, 12, 18, 0]],
        [datetime(2017, 2, 10, 0, 0, 0), [0, 6, 12, 18, 0]],
        [datetime(2016, 1, 1, 12, 30, 0), [12.5, 18.5, 0.5, 6.5, 12.5]],
    )
)
def test__local_time(time_gmt, expected):
    da_lon = xr.DataArray(
        [[0., 90., 180., -90., 360]],
        dims=["time", "x"],
        coords={"x": range(5), "time": [time_gmt]}
    )
    assert np.allclose(_local_time(da_lon, "time").values, expected)


def _generate_time_coords(time_of_day: tuple):
    h, m, s = time_of_day
    return datetime(2016, 1, 1, h, m, s)


@pytest.fixture
def ds(request):
    values, times_of_day = request.param
    time_coords = list(map(_generate_time_coords, times_of_day))
    da = xr.DataArray(
        [values for time in time_coords],
        dims=["time", "x"],
        coords={"x": [0, 1], "time": time_coords}
    ).rename("test_var")
    return xr.Dataset({"test_var": da})


@pytest.mark.parametrize("ds, diurnal_bin_means",
    (
        [[[1., 2.], [(0, 30, 0), (6, 30, 0)]], [1, 1, 2, 2]],
        [[[1., 2.], [(0, 30, 0), (0, 30, 0)]], [1., 0, 2., 0]],
        [[[1., 2.], [(0, 30, 0), (12, 30, 0)]], [1.5, 0, 1.5, 0]],
    ),
    indirect=["ds"]
)
def test_bin_diurnal_cycle(ds, diurnal_bin_means):
    da_lon = xr.DataArray(
        [0., 180.],
        dims=["x"],
        coords={"x": [0, 1]}
    )
    da_var = ds["test_var"]
    assert np.allclose(
        bin_diurnal_cycle(da_var, da_lon, ["test_var"], n_bins=4)["test_var"],
        diurnal_bin_means
    )
