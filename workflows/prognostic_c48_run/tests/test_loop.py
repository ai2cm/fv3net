import numpy as np
import xarray as xr

from runtime.loop import add_tendency

da = xr.DataArray(np.zeros((6, 4, 4)), dims=["z", "y", "x"], attrs={"units": "K"})
state = {"air_temperature": da, "specific_humidity": xr.full_like(da, 0.1)}
dt = 2.0


def test_add_tendency():
    tendency = {"dQ1": xr.full_like(da, 1.0)}
    updated_state = add_tendency(state, tendency, dt)
    expected_updated_state = {"air_temperature": xr.full_like(da, 2.0)}
    assert set(expected_updated_state) == set(updated_state)
    xr.testing.assert_identical(
        updated_state["air_temperature"], expected_updated_state["air_temperature"]
    )


def test_add_tendency_with_masked_values():
    tendency = {"dQ1": xr.full_like(da, 1.0).where(da.z > 3, np.nan)}
    updated_state = add_tendency(state, tendency, dt)
    expected_updated_state = {
        "air_temperature": xr.full_like(da, 2.0).where(da.z > 3, 0.0)
    }
    assert set(expected_updated_state) == set(updated_state)
    xr.testing.assert_identical(
        updated_state["air_temperature"], expected_updated_state["air_temperature"]
    )
