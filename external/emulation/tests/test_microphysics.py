from typing import Iterable

import numpy as np
import pytest
import tensorflow as tf
from emulation._emulate.microphysics import (
    MicrophysicsHook,
    NoModel,
    RenamedOutputModel,
    _load_tf_model,
    _unpack_predictions,
)


def test__unpack_predictions_single_out():
    data = np.arange(20).reshape(4, 5)

    out_names = ["field1"]

    result = _unpack_predictions(data, out_names)

    assert len(result) == 1
    assert result["field1"].shape == (5, 4)


def test__unpack_predictions_multi_out():
    data = np.arange(20).reshape(4, 5)

    out_names = ["field1", "field2", "field3"]

    result = _unpack_predictions([data] * 3, out_names)

    assert len(result) == len(out_names)
    for name in out_names:
        assert name in result

    for name in out_names:
        assert result[name].shape == (5, 4)


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


def test_NoModel():
    model = NoModel()

    in_ = model.input_names
    out_ = model.output_names
    pred = model.predict(1)

    for value in [in_, out_, pred]:
        assert not value
        assert isinstance(value, Iterable)


def test_load_tf_model_NoModel():
    model = _load_tf_model("NO_MODEL")
    assert isinstance(model, NoModel)


def test_microphysics_NoModel(dummy_rundir):

    state = {"empty_state": 1}
    hook = MicrophysicsHook("NO_MODEL")
    hook.microphysics(state)

    assert state == {"empty_state": 1}


def test_RenamedOutputModel():
    in_ = tf.keras.layers.Input(shape=(63,), name="air_temperature_input")
    old_names = ["a", "b"]
    new_names = ["c", "d"]
    out_ = [tf.keras.layers.Lambda(lambda x: x, name=name)(in_) for name in old_names]
    model = tf.keras.Model(inputs=in_, outputs=out_)
    renamed_model = RenamedOutputModel(model, dict(zip(old_names, new_names)))

    assert renamed_model.output_names == new_names
