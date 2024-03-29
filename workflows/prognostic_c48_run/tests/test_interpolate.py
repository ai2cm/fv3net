from runtime.interpolate import time_interpolate_func, label_to_time
import cftime
import xarray as xr
import numpy as np
from datetime import timedelta
import pytest


def test__label_to_time():
    time, label = cftime.DatetimeJulian(2015, 1, 20, 6, 30, 0), "20150120.063000"
    result = label_to_time(label)
    assert result == time


@pytest.mark.parametrize("fraction", [0, 0.25, 0.5, 0.75, 1])
def test_time_interpolate_func_has_correct_value(fraction):
    initial_time = cftime.DatetimeJulian(2016, 1, 1)
    frequency = timedelta(hours=3)
    attrs = {"units": "foo"}

    def func(time):
        value = float(time - initial_time > timedelta(hours=1.5))
        return {"a": xr.DataArray(data=np.array([value]), dims=["x"], attrs=attrs)}

    myfunc = time_interpolate_func(func, frequency, initial_time)
    ans = myfunc(initial_time + frequency * fraction)
    assert isinstance(ans["a"], xr.DataArray)
    assert float(ans["a"].values) == pytest.approx(fraction)
    assert ans["a"].attrs == attrs


def test_time_interpolate_func_only_grabs_correct_points():
    initial_time = cftime.DatetimeJulian(2016, 1, 1)
    frequency = timedelta(hours=2)

    valid_times = [
        initial_time,
        initial_time + frequency,
    ]

    def assert_passed_valid_times(time):
        assert time in valid_times
        return {}

    myfunc = time_interpolate_func(assert_passed_valid_times, frequency, initial_time)

    # will raise error if incorrect times grabbed
    myfunc(initial_time + frequency / 3)

    with pytest.raises(AssertionError):
        myfunc(initial_time + 4 * frequency / 3)
