import numpy as np
import xarray as xr

from runtime.loop import add_tendency


def test_add_tendency_fillna():
    state = {"air_temperature": xr.DataArray([0.0, 1.0, 2.0], dims=["x"])}
    tendency = {"dQ1": xr.DataArray([1.0, np.nan, 1.0], dims=["x"])}
    updated_state = add_tendency(state, tendency, 1.0)
    expected_after = {"air_temperature": xr.DataArray([1.0, 1.0, 3.0], dims=["x"])}
    for name, state in updated_state.items():
        xr.testing.assert_allclose(expected_after[name], state)
