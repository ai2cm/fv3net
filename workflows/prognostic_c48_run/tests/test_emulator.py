import tensorflow as tf
import joblib
from runtime.emulator import (
    PrognosticAdapter,
    Config,
)
from fv3fit.emulation.thermobasis.emulator import Config as MLConfig


def regression_state(state, regtest):
    for v in sorted(state):
        print(v, joblib.hash(state[v].values), file=regtest)


def test_state_regression(state, regtest):
    regression_state(state, regtest)


def test_adapter_regression(state, regtest):
    tf.random.set_seed(0)

    name = "add_one"

    emulate = PrognosticAdapter(
        Config(MLConfig(levels=state["air_temperature"].z.size)),
        state,
        diagnostic_variables={
            "emulator_latent_heat_flux",
            "tendency_of_specific_humidity_due_to_emulator",
            f"tendency_of_specific_humidity_due_to_{name}",
        },
        timestep=900,
    )

    def add_one_to_temperature():
        state["air_temperature"] += 1
        return {"some_diag": state["specific_humidity"]}

    out = emulate(name, add_one_to_temperature)()

    # sort to make the check deterministic
    regression_state(out, regtest)
