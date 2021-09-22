import xarray as xr
import tensorflow as tf
import joblib
from runtime.emulator import (
    PrognosticAdapter,
    Config,
    compute_mask_default,
    update_state_with_emulator,
    _get_mask_func,
    compute_mask_2021_09_16,
)
from fv3fit.emulation.thermobasis.emulator import Config as MLConfig


def testupdate_state_with_emulator(state):
    qv = "specific_humidity"
    old_state = state
    state = state.copy()

    z_slice = slice(0, 10)
    new = {qv: state[qv] + 1.0}

    update_state_with_emulator(
        state,
        new,
        compute_mask=lambda name, arr: xr.DataArray(name == qv) & (arr.z < 10),
    )

    xr.testing.assert_allclose(state[qv].sel(z=z_slice), old_state[qv].sel(z=z_slice))
    new_z_slice = slice(10, None)
    xr.testing.assert_allclose(state[qv].sel(z=new_z_slice), new[qv].sel(z=new_z_slice))


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
    for v in sorted(out):
        print(v, joblib.hash(out[v].values), file=regtest)


def test__get_mask_func_dynamics_lookup(state):
    for name in state:
        xr.testing.assert_equal(
            _get_mask_func(Config(mask_kind="2021_09_16"))(name, state[name]),
            compute_mask_2021_09_16(name, state[name]),
        )


def test__get_mask_func_default(state):
    for name in state:
        xr.testing.assert_equal(
            _get_mask_func(Config())(name, state[name]),
            compute_mask_default(name, state[name]),
        )
