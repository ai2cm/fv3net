import xarray as xr
from runtime.emulator.adapter import (
    update_state_with_emulator,
    _update_state_with_emulator,
)
import pytest


def test_update_state_with_emulator(state):
    qv = "specific_humidity"
    old_state = state
    state = state.copy()

    z_slice = slice(0, 10)
    new = {qv: state[qv] + 1.0}

    _update_state_with_emulator(
        state, new, from_orig=lambda name, arr: xr.DataArray(name == qv) & (arr.z < 10)
    )

    xr.testing.assert_allclose(state[qv].sel(z=z_slice), old_state[qv].sel(z=z_slice))
    new_z_slice = slice(10, None)
    xr.testing.assert_allclose(state[qv].sel(z=new_z_slice), new[qv].sel(z=new_z_slice))


@pytest.mark.parametrize("ignore_humidity_below", [0, 65, None])
def test_integration_with_from_orig(state, ignore_humidity_below):
    update_state_with_emulator(
        state, state, ignore_humidity_below=ignore_humidity_below
    )
