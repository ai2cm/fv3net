import numpy as np
import xarray as xr

from runtime.utils import ds_to_quantity_state, quantity_state_to_ds

da = xr.DataArray(np.ones((3, 4, 4)), dims=["tile", "x", "y"])
ds = xr.Dataset(
    {"temp": da.assign_attrs(units="K"), "sphum": da.assign_attrs(units="kg/kg")}
)


def test_ds_to_quantity_round_trip():
    round_tripped_ds = quantity_state_to_ds(ds_to_quantity_state(ds))
    xr.testing.assert_identical(ds, round_tripped_ds)
