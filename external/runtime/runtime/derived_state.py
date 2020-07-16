from datetime import datetime
from typing import Mapping

import vcm
import xarray as xr

import fv3util


class DerivedFV3State:
    """A uniform mapping-like interface to the FV3GFS model state
    
    This class provides two features

    1. wraps the fv3gfs getters with a Mapping interface, that always returns
       DataArray and has time as an attribute (since this isn't a DataArray).
       This insulates runfiles from the details of Quantity
       
    2. Register and computing derived variables transparently

    """

    _VARIABLES = {}

    def __init__(self, getter):
        """
        Args:
            getter: the fv3gfs object or a mock of it.
        """
        self._getter = getter

    @classmethod
    def register(cls, name: str):
        """Register a function as a derived variable

        See the cos_zenith_angle function below

        Args:
            name: the name the derived variable will be available under
        """

        def decorator(func):
            cls._VARIABLES[name] = func
            return func

        return decorator

    @property
    def time(self) -> datetime:
        return self._getter.get_state(["time"])["time"]

    def __getitem__(self, key: str) -> xr.Dataset:
        if key == "time":
            raise KeyError("To access time use the `time` property of this object.")

        if key in self._VARIABLES:
            return self._VARIABLES[key](self)
        else:
            return self._getter.get_state([key])[key].data_array

    def __setitem__(self, key: str, value: xr.DataArray):
        self._getter.set_state({key: fv3util.Quantity.from_data_array(value)})

    def update(self, items: Mapping[str, xr.DataArray]):
        """Update state from another mapping

        This may be faster than setting each item individually.
        
        Same as dict.update.
        """
        self._getter.set_state(
            {
                key: fv3util.Quantity.from_data_array(value)
                for key, value in items.items()
            }
        )


@DerivedFV3State.register("cos_zenith_angle")
def cos_zenith_angle(self):
    return xr.apply_ufunc(
        lambda lon, lat: vcm.cos_zenith_angle(self.time, lon, lat),
        self["longitude"],
        self["latitude"],
        dask="allowed",
    )
