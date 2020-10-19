from nudging_runfile import (
    implied_precipitation,
    total_precipitation,
    sst_from_reference,
    time_interpolate_func,
    time_to_label,
    label_to_time,
)
from datetime import timedelta
from fv3gfs.util import Quantity
import pytest
import xarray as xr
import numpy as np
import cftime


@pytest.fixture()
def xr_darray():
    return xr.DataArray(
        np.array([[2.0, 6.0], [6.0, 2.0]]), dims=["x", "z"], attrs={"units": "m"}
    )


@pytest.fixture()
def quantity(xr_darray):
    return Quantity.from_data_array(xr_darray)


@pytest.fixture()
def timestep():
    return timedelta(seconds=1)


def test_total_precipitation_positive(xr_darray, timestep):
    model_precip = 0.0 * xr_darray
    column_moistening = xr_darray
    total_precip = total_precipitation(model_precip, column_moistening, timestep)
    assert total_precip.min().values >= 0


def test_implied_precipitation(quantity, timestep):
    output = implied_precipitation(quantity, quantity, quantity, timestep)
    assert isinstance(output, np.ndarray)


def test_sst_set_to_reference(quantity):
    land_sea_mask = Quantity.from_data_array(
        xr.DataArray(np.array([0.0, 1.0, 2.0]), dims=["x"], attrs={"units": None})
    )
    reference_sfc_temp = Quantity.from_data_array(
        xr.DataArray(np.array([1.0, 1.0, 1.0]), dims=["x"], attrs={"units": "degK"})
    )
    model_sfc_temp = Quantity.from_data_array(
        xr.DataArray(np.array([-1.0, -1.0, -1.0]), dims=["x"], attrs={"units": "degK"})
    )
    assert np.allclose(
        sst_from_reference(reference_sfc_temp, model_sfc_temp, land_sea_mask),
        [1.0, -1.0, -1.0],
    )


@pytest.mark.parametrize("fraction", [0, 0.25, 0.5, 0.75, 1])
def test_time_interpolate_func_has_correct_value(fraction):
    initial_time = cftime.DatetimeJulian(2016, 1, 1)
    frequency = timedelta(hours=3)

    def func(time):
        value = float(time - initial_time > timedelta(hours=1.5))
        return {"a": Quantity(data=np.array([value]), dims=["x"], units="")}

    myfunc = time_interpolate_func(func, frequency, initial_time)
    ans = myfunc(initial_time + frequency * fraction)
    assert isinstance(ans["a"], Quantity)
    assert float(ans["a"].view[:]) == pytest.approx(fraction)


@pytest.mark.parametrize(
    "middle_time",
    [
        cftime.DatetimeJulian(2016, 1, 1, 5, 0, 0),
        cftime.DatetimeJulian(2016, 1, 1, 5, 15, 0),
    ],
)
def test_time_interpolate_func_has_time(middle_time):
    initial_time = cftime.DatetimeJulian(2016, 1, 1)

    def func(time):
        return {"a": Quantity(data=np.array([1.0]), dims=["x"], units="")}

    myfunc = time_interpolate_func(func, timedelta(hours=1), initial_time)
    ans = myfunc(middle_time)
    assert ans["time"] == middle_time


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


def test_time_to_label():
    time, label = cftime.DatetimeJulian(2015, 1, 20, 6, 30, 0), "20150120.063000"
    result = time_to_label(time)
    assert result == label


def test_label_to_time():
    time, label = cftime.DatetimeJulian(2015, 1, 20, 6, 30, 0), "20150120.063000"
    result = label_to_time(label)
    assert result == time
