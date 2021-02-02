from runtime.nudging import (
    _sst_from_reference,
    _time_interpolate_func,
    _time_to_label,
    _label_to_time,
    get_nudging_tendency,
)
import xarray as xr
from datetime import timedelta
import pytest
import numpy as np
import cftime
import copy


def test__sst_from_reference():
    land_sea_mask = xr.DataArray(
        np.array([0.0, 1.0, 2.0]), dims=["x"], attrs={"units": None}
    )
    reference_sfc_temp = xr.DataArray(
        np.array([1.0, 1.0, 1.0]), dims=["x"], attrs={"units": "degK"}
    )
    model_sfc_temp = xr.DataArray(
        np.array([-1.0, -1.0, -1.0]), dims=["x"], attrs={"units": "degK"}
    )
    assert np.allclose(
        _sst_from_reference(reference_sfc_temp, model_sfc_temp, land_sea_mask),
        [1.0, -1.0, -1.0],
    )


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


# tests of nudging tendency below adapted from fv3gfs.util versions


@pytest.fixture(params=["empty", "one_var", "two_vars"])
def state(request):
    if request.param == "empty":
        return {}
    elif request.param == "one_var":
        return {"var1": xr.DataArray(np.ones([5]), dims=["dim1"], attrs={"units": "m"})}
    elif request.param == "two_vars":
        return {
            "var1": xr.DataArray(np.ones([5]), dims=["dim1"], attrs={"units": "m"}),
            "var2": xr.DataArray(np.ones([5]), dims=["dim_2"], attrs={"units": "m"}),
        }
    else:
        raise NotImplementedError()


@pytest.fixture(params=["equal", "plus_one", "extra_var"])
def reference_difference(request):
    return request.param


@pytest.fixture
def reference_state(reference_difference, state):
    if reference_difference == "equal":
        reference_state = copy.deepcopy(state)
    elif reference_difference == "extra_var":
        reference_state = copy.deepcopy(state)
        reference_state["extra_var"] = xr.DataArray(
            np.ones([5]), dims=["dim1"], attrs={"units": "m"}
        )
    elif reference_difference == "plus_one":
        reference_state = copy.deepcopy(state)
        for array in reference_state.values():
            array.values[:] += 1.0
    else:
        raise NotImplementedError()
    return reference_state


@pytest.fixture(params=[0.1, 0.5, 1.0])
def multiple_of_timestep(request):
    return request.param


@pytest.fixture
def nudging_timescales(state, timestep, multiple_of_timestep):
    return_dict = {}
    for name in state.keys():
        return_dict[name] = timedelta(
            seconds=multiple_of_timestep * timestep.total_seconds()
        )
    return return_dict


@pytest.fixture
def nudging_tendencies(reference_difference, state, nudging_timescales):
    if reference_difference in ("equal", "extra_var"):
        tendencies = copy.deepcopy(state)
        for name, array in tendencies.items():
            array.values[:] = 0.0
            tendencies[name] = array.assign_attrs(
                {"units": array.attrs["units"] + " s^-1"}
            )
    elif reference_difference == "plus_one":
        tendencies = copy.deepcopy(state)
        for name, array in tendencies.items():
            array.data[:] = 1.0 / nudging_timescales[name].total_seconds()
            tendencies[name] = array.assign_attrs(
                {"units": array.attrs["units"] + " s^-1"}
            )
    else:
        raise NotImplementedError()
    return tendencies


@pytest.fixture(params=["one_second", "one_hour", "30_seconds"])
def timestep(request):
    if request.param == "one_second":
        return timedelta(seconds=1)
    if request.param == "one_hour":
        return timedelta(hours=1)
    if request.param == "30_seconds":
        return timedelta(seconds=30)
    else:
        raise NotImplementedError


def test_get_nudging_tendency(
    state, reference_state, nudging_timescales, nudging_tendencies
):
    result = get_nudging_tendency(state, reference_state, nudging_timescales)
    for name, tendency in nudging_tendencies.items():
        np.testing.assert_array_equal(result[name].data, tendency.data)
        assert result[name].dims == tendency.dims
        assert result[name].attrs["units"] == tendency.attrs["units"]
