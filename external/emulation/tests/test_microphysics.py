import datetime
import numpy as np

import cftime
import pytest
from emulation._emulate.microphysics import (
    MicrophysicsHook,
    always_emulator,
    IntervalSchedule,
    TimeMask,
)


def test_Config_integration(saved_model_path):

    config = MicrophysicsHook(saved_model_path, always_emulator)

    state = {
        "air_temperature_input": np.ones((63, 100)),
    }

    # something that will be overwritten by call to microphysics
    state["air_temperature_dummy"] = state["air_temperature_input"]

    for i in range(3):
        input = state["air_temperature_input"]

        config.microphysics(state)

        # microphysics saves any key overwrites as a diagnostic
        updated = state["air_temperature_dummy"]
        diag = state["air_temperature_dummy_physics_diag"]

        np.testing.assert_array_equal(diag, input)
        np.testing.assert_array_almost_equal(input + 1, updated)

        state["air_temperature_input"] = updated


def test_IntervalSchedule():
    scheduler = IntervalSchedule(
        datetime.timedelta(hours=3), cftime.DatetimeJulian(2000, 1, 1)
    )
    assert scheduler(cftime.DatetimeJulian(2000, 1, 1)) == 1
    assert scheduler(cftime.DatetimeJulian(2000, 1, 1, 1)) == 1
    assert scheduler(cftime.DatetimeJulian(2000, 1, 1, 1, 30)) == 0
    assert scheduler(cftime.DatetimeJulian(2000, 1, 1, 2)) == 0
    assert scheduler(cftime.DatetimeJulian(2000, 1, 20)) == 1


@pytest.mark.parametrize("weight", [0.0, 0.5, 1.0])
def test_TimeMask(weight):
    expected = 1 - weight
    mask = TimeMask(schedule=lambda time: weight)
    left = {"a": 0.0, "model_time": [2021, 1, 1, 0, 0, 0]}
    right = {"a": 1.0}
    assert mask(left, right) == {"a": expected}
