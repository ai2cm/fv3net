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
        {"time": ([], time), "lat": (["x"], [lat]), "lon": (["x"], [lon])}
    )
    expected = cos_zenith_angle(time, lon, lat)
    ans = cos_zenith_angle(dataset.time, dataset.lon, dataset.lat)
    assert isinstance(ans, xr.DataArray)
    assert ans.item() == pytest.approx(expected)
    assert ans.name == "cos_zenith_angle"


def _dataset(lon_units, lat_units):
    time = DatetimeJulian(2020, 3, 21, 12, 0, 0)
    lat = 10
    lon = 10
    if lon_units == "radians":
        lon = np.deg2rad(lon)
    if lat_units == "radians":
        lat = np.deg2rad(lat)
    lon_attrs = {"units": lon_units}
    lat_attrs = {"units": lat_units}
    return xr.Dataset(
        {
            "time": ([], time),
            "lat": (["x"], [lat], lat_attrs),
            "lon": (["x"], [lon], lon_attrs),
        }
    )


@pytest.mark.parametrize("lon_units", ["degrees", "radians"])
@pytest.mark.parametrize("lat_units", ["degrees", "radians"])
def test_cos_zenith_angle_dataarray_converts_units(lon_units, lat_units):
    ds_in_degrees = _dataset("degrees", "degrees")
    expected = cos_zenith_angle(
        ds_in_degrees.time, ds_in_degrees.lon, ds_in_degrees.lat
    )

    ds_with_test_units = _dataset(lon_units, lat_units)
    result = cos_zenith_angle(
        ds_with_test_units.time, ds_with_test_units.lon, ds_with_test_units.lat
    )

    xr.testing.assert_identical(result, expected)
