import datetime
import numpy as np
import pytest
import tensorflow as tf

import cftime
from emulation._emulate.microphysics import (
    MicrophysicsHook,
    IntervalSchedule,
    TimeMask,
)


def _create_model():
    in_ = tf.keras.layers.Input(shape=(63,), name="air_temperature_input")
    out_ = tf.keras.layers.Lambda(lambda x: x + 1, name="air_temperature_dummy")(in_)
    model = tf.keras.Model(inputs=in_, outputs=out_)
    return model


def _create_model_dict():
    in_ = tf.keras.layers.Input(shape=(63,), name="air_temperature_input")
    out_ = tf.keras.layers.Lambda(lambda x: x + 1)(in_)
    model = tf.keras.Model(inputs=in_, outputs={"air_temperature_dummy": out_})
    return model


@pytest.mark.parametrize("model_factory", [_create_model, _create_model_dict])
def test_Config_integration(model_factory):
    hook = MicrophysicsHook(model_factory(), (lambda x, y, z: z))
    n = 100
    state = {
        "air_temperature_input": np.ones((63, n)),
        "latitude": np.linspace(-60, 60, n),
    }

    # something that will be overwritten by call to microphysics
    state["air_temperature_dummy"] = state["air_temperature_input"]

    for i in range(3):
        input = state["air_temperature_input"]

        hook.microphysics(state)

        # microphysics saves any key overwrites as a diagnostic
        updated = state["air_temperature_dummy"]
        diag = state["air_temperature_dummy_physics_diag"]

        np.testing.assert_array_equal(diag, input)
        np.testing.assert_array_almost_equal(input + 1, updated)

        state["air_temperature_input"] = updated


def test_MicrophysicsHook_model_with_new_output_name():
    """Test a bug that happens when the ML model predicts an output not present
    in the input state
    """
    n = 3
    c = 4
    in_ = tf.keras.layers.Input(shape=(c,), name="in")
    out_ = tf.keras.layers.Lambda(lambda x: x + 1, name="out")(in_)
    model = tf.keras.Model(inputs=in_, outputs=[out_])

    hook = MicrophysicsHook(model, (lambda x, y, z: z))

    state = {
        "in": np.ones((c, n)),
        "latitude": np.linspace(-60, 60, n),
    }

    assert "out" not in state
    hook.microphysics(state)


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
