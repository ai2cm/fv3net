import pytest
import numpy as np
import tensorflow as tf

from emulation._emulate.microphysics import (
    _unpack_predictions,
    MicrophysicsConfig
)


def create_model():
    in_ = tf.keras.layers.Input(shape=(63,), name="air_temperature_input")
    out_ = tf.keras.layers.Lambda(lambda x: x + 1, name="air_temperature_dummy")(in_)
    model = tf.keras.Model(inputs=in_, outputs=out_)

    return model


@pytest.fixture()
def saved_model_path(tmp_path):
    model = create_model()
    path = tmp_path / "dummy_model.tf"
    model.save(path.as_posix(), save_format="tf")

    return str(path)


def test__unpack_predictions_single_out():
    data = np.arange(20).reshape(4, 5)

    out_names = ["field1"]

    result = _unpack_predictions(data, out_names)

    assert len(result) == 1
    assert result["field1"].shape == (5, 4)


def test__unpack_predictions_multi_out():
    data = np.arange(20).reshape(4, 5)

    out_names = ["field1", "field2", "field3"]

    result = _unpack_predictions([data]*3, out_names)

    assert len(result) == len(out_names)
    for name in out_names:
        assert name in result

    for name in out_names:
        assert result[name].shape == (5, 4)


def test_MicrophysicsConfig_integration(saved_model_path, dummy_rundir):

    config = MicrophysicsConfig(saved_model_path)

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
