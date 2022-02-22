import numpy as np
import xarray as xr

from runtime.loop import add_tendency


def test_add_tendency_fillna():
    state = {"air_temperature": xr.DataArray([0.0, 1.0, 2.0], dims=["z"])}
    tendency = {"dQ1": xr.DataArray([1.0, np.nan, 1.0], dims=["z"])}
    updated_state, tendency_filled_frac = add_tendency(state, tendency, 1.0)
    expected_after = {"air_temperature": xr.DataArray([1.0, 1.0, 3.0], dims=["z"])}
    for name, state in updated_state.items():
        xr.testing.assert_allclose(expected_after[name], state)
    assert tendency_filled_frac["dQ1_filled_frac"].values.item() == 1.0 / 3.0
