from radiation.wrapper_api import _get_forecast_time_index, _is_compute_timestep
import cftime
import pytest

INIT_TIME = cftime.DatetimeJulian(2000, 1, 1, 0, 0, 0, 0)


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
