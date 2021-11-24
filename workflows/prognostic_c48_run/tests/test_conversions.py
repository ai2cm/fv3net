import numpy as np
import xarray as xr

from runtime.conversions import dataset_to_quantity_state, quantity_state_to_dataset

da = xr.DataArray(np.ones((3, 4, 4)), dims=["tile", "x", "y"])
ds = xr.Dataset(
    {"temp": da.assign_attrs(units="K"), "sphum": da.assign_attrs(units="kg/kg")}
)
ds_no_units = xr.Dataset({"temp": da, "sphum": da})


def test_ds_to_quantity_round_trip():
    round_tripped_ds = quantity_state_to_dataset(dataset_to_quantity_state(ds))
    xr.testing.assert_identical(ds, round_tripped_ds)


def test_ds_to_quantity_no_units():
    dataset_to_quantity_state(ds_no_units)
