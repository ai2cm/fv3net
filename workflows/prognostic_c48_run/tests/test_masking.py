import xarray as xr
from runtime.masking import (
    compute_mask_2021_09_16,
    compute_mask_default,
    get_mask,
    where_masked,
)


def testupdate_state_with_emulator(state):
    qv = "specific_humidity"
    old_state = state

    z_slice = slice(0, 10)
    new = {qv: state[qv] + 1.0}

    blended_state = where_masked(
        state,
        new,
        compute_mask=lambda name, arr: xr.DataArray(name == qv) & (arr.z < 10),
    )

    xr.testing.assert_allclose(
        blended_state[qv].sel(z=z_slice), old_state[qv].sel(z=z_slice)
    )
    new_z_slice = slice(10, None)
    xr.testing.assert_allclose(
        blended_state[qv].sel(z=new_z_slice), new[qv].sel(z=new_z_slice)
    )


def test__get_mask_func_default(state):
    for name in state:
        xr.testing.assert_equal(
            get_mask(kind="default")(name, state[name]),
            compute_mask_default(name, state[name]),
        )


def test__get_mask_func_dynamics_lookup(state):
    for name in state:
        xr.testing.assert_equal(
            get_mask(kind="2021_09_16")(name, state[name]),
            compute_mask_2021_09_16(name, state[name]),
        )
