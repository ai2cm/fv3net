import pytest
import numpy as np
import tensorflow as tf

from emulation._emulate.microphysics import (
    MicrophysicsHook,
    RenamedOutputModel,
)


def test_Config_integration(saved_model_path, dummy_rundir):

    config = MicrophysicsHook(saved_model_path)

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


def test_error_on_call():

    with pytest.raises(ImportError):
        from emulation import microphysics

        microphysics({})


def test_RenamedOutputModel():
    in_ = tf.keras.layers.Input(shape=(63,), name="air_temperature_input")
    old_names = ["a", "b"]
    new_names = ["c", "d"]
    out_ = [tf.keras.layers.Lambda(lambda x: x, name=name)(in_) for name in old_names]
    model = tf.keras.Model(inputs=in_, outputs=out_)
    renamed_model = RenamedOutputModel(model, dict(zip(old_names, new_names)))

    assert renamed_model.output_names == new_names
