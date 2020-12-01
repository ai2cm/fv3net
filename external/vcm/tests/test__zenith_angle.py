from cftime import DatetimeJulian, DatetimeNoLeap
from datetime import datetime
import numpy as np
import pytest
import xarray as xr

from vcm import cos_zenith_angle


@pytest.mark.parametrize(
    "time, lon, lat, expected",
    (
        [DatetimeJulian(2020, 3, 21, 12, 0, 0), 0.0, 0.0, 1.0],
        [DatetimeJulian(2020, 3, 21, 18, 0, 0), -90.0, 0.0, 1.0],
        [DatetimeJulian(2020, 3, 21, 18, 0, 0), 270.0, 0.0, 1.0],
        [DatetimeJulian(2020, 7, 6, 12, 0, 0), -90.0, 0.0, -0.0196310],
        [DatetimeJulian(2020, 7, 6, 9, 0, 0), 40.0, 40.0, 0.9501915],
        [DatetimeJulian(2020, 7, 6, 12, 0, 0), 0.0, 90.0, 0.3843733],
        [datetime(2020, 3, 21, 12, 0, 0), 0.0, 0.0, 1.0],
        [datetime(2020, 3, 21, 18, 0, 0), -90.0, 0.0, 1.0],
        [datetime(2020, 3, 21, 18, 0, 0), 270.0, 0.0, 1.0],
        [datetime(2020, 7, 6, 12, 0, 0), -90.0, 0.0, -0.0196310],
        [datetime(2020, 7, 6, 9, 0, 0), 40.0, 40.0, 0.9501915],
        [datetime(2020, 7, 6, 12, 0, 0), 0.0, 90.0, 0.3843733],
    ),
)
def test__sun_zenith_angle(time, lon, lat, expected):
    assert cos_zenith_angle(time, lon, lat) == pytest.approx(expected, abs=1e-3)


@pytest.mark.parametrize(
    "invalid_time",
    (
        DatetimeNoLeap(2000, 1, 1),
        np.array([DatetimeNoLeap(2000, 1, 1), DatetimeNoLeap(2000, 2, 1)]),
    ),
    ids=["scalar", "array"],
)
def test__sun_zenith_angle_invalid_time(invalid_time):
    with pytest.raises(ValueError, match="model_time has an invalid date type"):
        cos_zenith_angle(invalid_time, 0.0, 0.0)


def test_cos_zenith_angle_dataarray():
    time = DatetimeJulian(2020, 3, 21, 12, 0, 0)
    lat = 0
    lon = 0
    dataset = xr.Dataset(
        {"time": ([], time), "lat": (["x"], [lat]), "lon": (["x"], [lon]),}
    )
    expected = cos_zenith_angle(time, lon, lat)
    ans = cos_zenith_angle(dataset.time, dataset.lon, dataset.lat)
    assert isinstance(ans, xr.DataArray)
    assert ans.item() == pytest.approx(expected)
    assert ans.name == "cos_zenith_angle"
