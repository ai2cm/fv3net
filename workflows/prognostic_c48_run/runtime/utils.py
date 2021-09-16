import xarray as xr
import fv3gfs.util

from runtime.types import QuantityState


def ds_to_quantity_state(ds: xr.Dataset) -> QuantityState:
    quantity_state: QuantityState = {
        variable: fv3gfs.util.Quantity.from_data_array(ds[variable])
        for variable in ds.data_vars
    }
    return quantity_state


def quantity_state_to_ds(quantity_state: QuantityState) -> xr.Dataset:
    ds = xr.Dataset(
        {
            variable: quantity_state[variable].data_array
            for variable in quantity_state.keys()
        }
    )
    return ds
