import cftime
import numpy as np
from typing import Mapping, MutableMapping, Hashable
from toolz import dissoc
import xarray as xr

import fv3gfs.util
from vcm import DerivedMapping
from runtime.names import DELP

import fv3gfs.wrapper._properties
import fv3gfs.wrapper


class FV3StateMapper(Mapping):
    """ A mapping interface for the FV3GFS getter.
        
    Maps variables to the common names used in shared functions.
    By default adds mapping {"lon": "longitude", "lat": "latitude"}
    """

    def __init__(self, getter, alternate_keys: Mapping[str, str] = None):
        self._getter = getter
        self._alternate_keys = alternate_keys or {"lon": "longitude", "lat": "latitude"}

    def __getitem__(self, key: str) -> xr.DataArray:
        if key == "time":
            time = self._getter.get_state(["time"])["time"]
            return xr.DataArray(time, name="time")
        elif key == "latent_heat_flux":
            return self._getter.get_diagnostic_by_name("lhtfl").data_array
        elif key == "total_water":
            return self._total_water()
        elif key in ["lon", "lat"]:
            alternate_key = self._alternate_keys[key]
            return np.rad2deg(self._getter.get_state([alternate_key])[alternate_key].data_array)
        else:
            if key in self._alternate_keys:
                key = self._alternate_keys[key]
            return self._getter.get_state([key])[key].data_array

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(self.keys())

    def keys(self):
        dynamics_names = set(
            v["name"] for v in fv3gfs.wrapper._properties.DYNAMICS_PROPERTIES
        )
        physics_names = set(
            v["name"] for v in fv3gfs.wrapper._properties.PHYSICS_PROPERTIES
        )
        tracer_names = set(v for v in fv3gfs.wrapper.get_tracer_metadata())
        # see __getitem__
        local_names = {"latent_heat_flux", "total_water"}
        return dynamics_names | physics_names | tracer_names | local_names

    def _total_water(self):
        a = self._getter.get_tracer_metadata()
        water_species = [name for name in a if a[name]["is_water"]]
        return sum(self[name] for name in water_species)


class DerivedFV3State(MutableMapping):
    """A uniform mapping-like interface to the FV3GFS model state
    
    This class wraps the fv3gfs getters with the FV3StateMapper, that always returns
    DataArray and has time as an attribute (since this isn't a DataArray).
    
    This insulates runfiles from the details of Quantity
    
    """

    def __init__(self, getter):
        """
        Args:
            getter: the fv3gfs object or a mock of it.
        """
        self._getter = getter
        self._mapper = DerivedMapping(FV3StateMapper(getter, alternate_keys=None))

    @property
    def time(self) -> cftime.DatetimeJulian:
        return self._getter.get_state(["time"])["time"]

    def __getitem__(self, key: Hashable) -> xr.DataArray:
        return self._mapper[key]

    def __setitem__(self, key: str, value: xr.DataArray):
        self._getter.set_state_mass_conserving(
            {key: fv3gfs.util.Quantity.from_data_array(value)}
        )

    def keys(self):
        return self._mapper.keys()

    def update_mass_conserving(
        self, items: Mapping[Hashable, xr.DataArray],
    ):
        """Update state from another mapping

        This may be faster than setting each item individually. Same as dict.update.
        
        All states except for pressure thicknesses are set in a mass-conserving fashion.
        """
        if DELP in items:
            self._getter.set_state(
                {DELP: fv3gfs.util.Quantity.from_data_array(items[DELP])}
            )

        not_pressure = dissoc(items, DELP)
        self._getter.set_state_mass_conserving(
            {
                key: fv3gfs.util.Quantity.from_data_array(value)
                for key, value in not_pressure.items()
            }
        )

    def __delitem__(self):
        raise NotImplementedError()

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(self.keys())
