from runtime.nudging import (
    _time_interpolate_func,
    _time_to_label,
    _label_to_time,
)
import xarray as xr
from datetime import timedelta
import pytest
import numpy as np
import cftime


@pytest.mark.parametrize("fraction", [0, 0.25, 0.5, 0.75, 1])
def test__time_interpolate_func_has_correct_value(fraction):
    initial_time = cftime.DatetimeJulian(2016, 1, 1)
    frequency = timedelta(hours=3)

    def func(time):
        value = float(time - initial_time > timedelta(hours=1.5))
        return {"a": xr.DataArray(data=np.array([value]), dims=["x"])}

    myfunc = _time_interpolate_func(func, frequency, initial_time)
    ans = myfunc(initial_time + frequency * fraction)
    assert isinstance(ans["a"], xr.DataArray)
    assert float(ans["a"].values) == pytest.approx(fraction)


@pytest.mark.parametrize(
    "middle_time",
    [
        cftime.DatetimeJulian(2016, 1, 1, 5, 0, 0),
        cftime.DatetimeJulian(2016, 1, 1, 5, 15, 0),
    ],
)
def test__time_interpolate_func_has_time(middle_time):
    initial_time = cftime.DatetimeJulian(2016, 1, 1)

    def func(time):
        return {"a": xr.DataArray(data=np.array([1.0]), dims=["x"])}

    myfunc = _time_interpolate_func(func, timedelta(hours=1), initial_time)
    ans = myfunc(middle_time)
    assert ans["time"] == middle_time


def test__time_interpolate_func_only_grabs_correct_points():
    initial_time = cftime.DatetimeJulian(2016, 1, 1)
    frequency = timedelta(hours=2)

    valid_times = [
        initial_time,
        initial_time + frequency,
    ]

    def assert_passed_valid_times(time):
        assert time in valid_times
        return {}

    myfunc = _time_interpolate_func(assert_passed_valid_times, frequency, initial_time)

    # will raise error if incorrect times grabbed
    myfunc(initial_time + frequency / 3)

    with pytest.raises(AssertionError):
        myfunc(initial_time + 4 * frequency / 3)


def test__time_to_label():
    time, label = cftime.DatetimeJulian(2015, 1, 20, 6, 30, 0), "20150120.063000"
    result = _time_to_label(time)
    assert result == label


def test__label_to_time():
    time, label = cftime.DatetimeJulian(2015, 1, 20, 6, 30, 0), "20150120.063000"
    result = _label_to_time(label)
    assert result == time
