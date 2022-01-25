import numpy as np

from emulation._emulate.microphysics import MicrophysicsHook


def test_Config_integration(saved_model_path):

    config = MicrophysicsHook(saved_model_path, (lambda x, y, z: z))

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
