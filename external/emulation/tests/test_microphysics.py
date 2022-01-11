import pytest
import numpy as np
from typing import Iterable

from emulation._emulate.microphysics import (
    MicrophysicsHook,
    NoModel,
    _load_tf_model,
)


def test_Config_integration(saved_model_path, dummy_rundir):

    config = MicrophysicsHook(saved_model_path)

    n = 100
    state = {
        "air_temperature_input": np.ones((63, n)),
        "latitude": np.linspace(-60, 60, n),
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
