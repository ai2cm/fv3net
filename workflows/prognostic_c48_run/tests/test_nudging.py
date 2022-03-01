from runtime.names import STATE_NAME_TO_TENDENCY, TENDENCY_TO_STATE_NAME
from runtime.nudging import (
    _time_to_label,
    get_nudging_tendency,
    _rename_local_restarts,
)
from runtime.nudging import RestartCategoriesConfig
import xarray as xr
from datetime import timedelta
import pytest
import numpy as np
import cftime
import copy
import pathlib


def test__time_to_label():
    time, label = cftime.DatetimeJulian(2015, 1, 20, 6, 30, 0), "20150120.063000"
    result = _time_to_label(time)
    assert result == label


# tests of nudging tendency below adapted from pace.util versions


@pytest.fixture(params=["empty", "one_var", "multiple_vars"])
def state(request):
    if request.param == "empty":
        return {}
    elif request.param == "one_var":
        return {
            "air_temperature": xr.DataArray(
                np.ones([5]), dims=["dim1"], attrs={"units": "K"}
            )
        }
    elif request.param == "multiple_vars":
        return {
            "air_temperature": xr.DataArray(
                np.ones([5]), dims=["dim1"], attrs={"units": "K"}
            ),
            "specific_humidity": xr.DataArray(
                np.ones([5]), dims=["dim_2"], attrs={"units": "kg/kg"}
            ),
            "pressure_thickness_of_atmospheric_layer": xr.DataArray(
                np.ones([5]), dims=["dim_2"], attrs={"units": "Pa"}
            ),
            "x_wind": xr.DataArray(
                np.ones([5]), dims=["dim_2"], attrs={"units": "m/s"}
            ),
            "y_wind": xr.DataArray(
                np.ones([5]), dims=["dim_2"], attrs={"units": "m/s"}
            ),
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
        tendencies = {
            STATE_NAME_TO_TENDENCY[name]: field
            for name, field in copy.deepcopy(state).items()
        }
        for name, array in tendencies.items():
            array.values[:] = 0.0
            tendencies[name] = array.assign_attrs(
                {"units": array.attrs["units"] + " s^-1"}
            )
    elif reference_difference == "plus_one":
        tendencies = {
            STATE_NAME_TO_TENDENCY[name]: field
            for name, field in copy.deepcopy(state).items()
        }
        for name, array in tendencies.items():
            state_name = TENDENCY_TO_STATE_NAME[name]
            array.data[:] = 1.0 / nudging_timescales[state_name].total_seconds()
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


RESTART_CATEGORIES = {
    "core": "fv_core_coarse.res",
    "surface": "sfc_data_coarse",
    "tracer": "fv_tracer_coarse.res",
    "surface_wind": "fv_srf_wnd_coarse.res",
}
TIMESTAMP = "20160801.000000"


@pytest.fixture()
def restart_dir(tmp_path):
    sub = tmp_path / TIMESTAMP
    sub.mkdir()
    for specified_category in RESTART_CATEGORIES.values():
        pathlib.Path(
            sub / ".".join([TIMESTAMP, specified_category, "tile1.nc"])
        ).touch()
    return sub


def test__rename_local_restarts(restart_dir):
    restarts_config = RestartCategoriesConfig(**RESTART_CATEGORIES)
    _rename_local_restarts(restart_dir.as_posix(), restarts_config)
    renamed_files = sorted([file.name for file in restart_dir.iterdir()])
    standard_restarts_config = RestartCategoriesConfig()
    standard_categories = [
        getattr(standard_restarts_config, category)
        for category in vars(standard_restarts_config)
    ]
    intended_files = sorted(
        [
            ".".join([TIMESTAMP, standard_category, "tile1.nc"])
            for standard_category in standard_categories
        ]
    )
    assert renamed_files == intended_files
