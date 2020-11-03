from typing import Mapping, Hashable
import xarray as xr

import fv3gfs.util
from .base import DerivedMapping


class FV3StateMapper:
    """ A mapping interface for the FV3GFS getter.
    
    Allows the DerivedFV3State to work with the base
    DerivedMapping's __getitem__.
    
    Maps variables to the common names used in shared functions.
    By default adds mapping {"lon": "longitude", "lat": "latitude"}
    """

    def __init__(self, getter, alternate_keys: Mapping[str, str] = None):
        self._getter = getter
        self._alternate_keys = alternate_keys or {"lon": "longitude", "lat": "latitude"}

    def __getitem__(self, key: Hashable) -> xr.DataArray:
        if key == "time":
            return self._getter.get_state(["time"])["time"]
        else:
            if key in self._alternate_keys:
                key = self._alternate_keys[key]
            return self._getter.get_state([key])[key].data_array


class DerivedFV3State(DerivedMapping):
    """A uniform mapping-like interface to the FV3GFS model state
    
    This class provides two features

    1. wraps the fv3gfs getters with the FV3StateMapper, that always returns
       DataArray and has time as an attribute (since this isn't a DataArray).
       This insulates runfiles from the details of Quantity
       
    2. Register and computing derived variables transparently

    """

    def __init__(self, getter):
        """
        Args:
            getter: the fv3gfs object or a mock of it.
        """
        self._getter = getter
        self._mapper = FV3StateMapper(getter, alternate_keys=None)

    def __setitem__(self, key: str, value: xr.DataArray):
        self._getter.set_state_mass_conserving(
            {key: fv3gfs.util.Quantity.from_data_array(value)}
        )

    def update(self, items: Mapping[Hashable, xr.DataArray]):
        """Update state from another mapping

        This may be faster than setting each item individually.
        
        Same as dict.update.
        """
        self._getter.set_state_mass_conserving(
            {
                key: fv3gfs.util.Quantity.from_data_array(value)
                for key, value in items.items()
            }
        )


@DerivedFV3State.register("latent_heat_flux")
def latent_heat_flux(self):
    return self._getter.get_diagnostic_by_name("lhtfl").data_array


@DerivedFV3State.register("total_water")
def total_water(self):
    a = self._getter.get_tracer_metadata()
    water_species = [name for name in a if a[name]["is_water"]]
    return sum(self[name] for name in water_species)
