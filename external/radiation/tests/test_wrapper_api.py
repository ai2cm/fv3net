from radiation.wrapper_api import (
    _get_forecast_time_index,
    _is_compute_timestep,
    _solar_hour,
    Radiation,
    GFSPhysicsControlConfig,
)
from radiation.config import RadiationConfig
import numpy as np
import cftime
import datetime
import pandas as pd
import xarray as xr
import pytest

INIT_TIME = cftime.DatetimeJulian(2000, 1, 1, 0, 0, 0, 0)
INIT_TIME_OFF_HOUR = cftime.DatetimeJulian(2000, 1, 1, 0, 30, 0, 0)


@pytest.mark.parametrize(
    ["time", "init_time", "timestep_seconds", "expected_index"],
    [
        pytest.param(INIT_TIME, INIT_TIME, 900, 1, id="initial_forecast"),
        pytest.param(
            cftime.DatetimeJulian(2000, 1, 1, 0, 15, 0, 0),
            INIT_TIME,
            900,
            2,
            id="second_forecast",
        ),
        pytest.param(
            cftime.DatetimeJulian(2000, 1, 1, 0, 15, 0, 0),
            INIT_TIME,
            450,
            3,
            id="third_forecast",
        ),
        pytest.param(
            cftime.DatetimeJulian(2000, 1, 2, 0, 0, 0, 0),
            INIT_TIME,
            900,
            97,
            id="1_day_in",
        ),
    ],
)
def test__get_forecast_time_index(time, init_time, timestep_seconds, expected_index):
    index = _get_forecast_time_index(time, init_time, timestep_seconds)
    assert index == expected_index


@pytest.mark.parametrize(
    ["time_index", "compute_period", "expected_compute"],
    [
        pytest.param(1, 100, True, id="time_index_1"),
        pytest.param(2, 100, False, id="time_index_2"),
        pytest.param(101, 100, True, id="time_index_one_greater"),
        pytest.param(102, 1, True, id="compute_every_time"),
    ],
)
def test__is_compute_timestep(time_index, compute_period, expected_compute):
    compute = _is_compute_timestep(time_index, compute_period)
    assert compute == expected_compute


@pytest.mark.parametrize(
    ["time", "init_time", "expected_solar_hour"],
    [
        pytest.param(
            cftime.DatetimeJulian(2000, 1, 1, 0, 30, 0, 0),
            INIT_TIME,
            0.5,
            id="sub_hour",
        ),
        pytest.param(
            cftime.DatetimeJulian(2000, 1, 1, 1, 0, 0, 0), INIT_TIME, 1.0, id="one_hour"
        ),
        pytest.param(
            cftime.DatetimeJulian(2000, 1, 2, 0, 0, 0, 0), INIT_TIME, 0.0, id="one_day"
        ),
        pytest.param(
            cftime.DatetimeJulian(2000, 1, 1, 1, 0, 0, 0),
            INIT_TIME_OFF_HOUR,
            0.5,
            id="off_hour_init",
        ),
    ],
)
def test__solar_hour(time, init_time, expected_solar_hour):
    solar_hour = _solar_hour(time, init_time)
    assert np.isclose(solar_hour, expected_solar_hour)


class DummyComm:
    rank: int = 0

    def __init__(self):
        pass


RADIATION_KWARGS = {
    "tracer_inds": {
        "specific_humidity": 1,
        "cloud_water_mixing_ratio": 2,
        "rain_mixing_ratio": 3,
        "cloud_ice_mixing_ratio": 4,
        "snow_mixing_ratio": 5,
        "graupel_mixing_ratio": 6,
        "ozone_mixing_ratio": 7,
        "cloud_amount": 8,
    },
    "comm": DummyComm(),
    "timestep": 900.0,
    "init_time": datetime.datetime(2016, 8, 1, 0, 0, 0),
}


@pytest.mark.parametrize(
    ["overlap_option"],
    [
        pytest.param(None, id="default"),
        pytest.param(0, id="random_overlap"),
        pytest.param(1, id="max_random_overlap"),
        pytest.param(3, id="decorrelation_length_overlap"),
    ],
)
def test_overlap_options(overlap_option, regtest):
    if overlap_option is not None:
        rad_config = RadiationConfig(iovrlw=overlap_option, iovrsw=overlap_option)
    else:
        rad_config = RadiationConfig()
    with regtest:
        Radiation(rad_config, **RADIATION_KWARGS).validate()


def test_unimplemented_overlap_raises():
    overlap_option = 2
    rad_config = RadiationConfig(iovrlw=overlap_option, iovrsw=overlap_option)
    with pytest.raises(ValueError):
        Radiation(rad_config, **RADIATION_KWARGS).validate()


class MockRadiationDriver:
    def __init__(self):
        self.counter = 0

    def validate(self):
        pass

    def __call__(self):
        self.counter += 1
        return {"a": xr.DataArray(self.counter)}


class MockRadiation(Radiation):
    def _compute_radiation(self, time, state):
        return self._driver()


@pytest.mark.parametrize(
    ["radiation_timestep"],
    [
        pytest.param(900.0, id="same_as_physics_timestep"),
        pytest.param(1800.0, id="2x_physics_timestep"),
        pytest.param(3600.0, id="4x_physics_timestep"),
    ],
)
def test_wrapper_radiation_timestepping(radiation_timestep):
    radiation_driver = MockRadiationDriver()
    gfs_physics_control_config = GFSPhysicsControlConfig(
        fhswr=radiation_timestep, fhlwr=radiation_timestep
    )
    radiation_config = RadiationConfig(
        gfs_physics_control_config=gfs_physics_control_config
    )
    radiation_wrapper = MockRadiation(
        radiation_config, driver=radiation_driver, **RADIATION_KWARGS
    )
    physics_timestep = RADIATION_KWARGS["timestep"]
    init_time = RADIATION_KWARGS["init_time"]
    end_time = init_time + datetime.timedelta(seconds=radiation_timestep)
    for time in pd.date_range(
        init_time, end_time, freq=datetime.timedelta(seconds=physics_timestep)
    ):
        diags = radiation_wrapper(time, xr.Dataset())
        if time < end_time:
            assert diags["a"].item() == 1
        else:
            assert diags["a"].item() == 2
