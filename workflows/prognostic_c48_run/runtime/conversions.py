import xarray as xr
import pace.util

from runtime.types import QuantityState


def _insert_units_if_necessary(da: xr.DataArray) -> xr.DataArray:
    if "units" not in da.attrs:
        return da.assign_attrs(units="")
    else:
        return da


def dataset_to_quantity_state(ds: xr.Dataset) -> QuantityState:
    quantity_state: QuantityState = {
        variable: pace.util.Quantity.from_data_array(
            _insert_units_if_necessary(ds[variable])
        )
        for variable in ds.data_vars
    }
    return quantity_state


def quantity_state_to_dataset(quantity_state: QuantityState) -> xr.Dataset:
    ds = xr.Dataset(
        {
            variable: quantity_state[variable].data_array
            for variable in quantity_state.keys()
        }
    )
    return ds
