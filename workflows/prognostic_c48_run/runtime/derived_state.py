from typing import Hashable, Mapping, MutableMapping, Set

import cftime
import pace.util
import fv3gfs.wrapper
import fv3gfs.wrapper._properties
import numpy as np
import xarray as xr
from runtime.names import DELP, PHYSICS_PRECIP_RATE
from runtime.types import State
from toolz import dissoc
from vcm import DerivedMapping, round_time


class FV3StateMapper(Mapping):
    """ A mapping interface for the FV3GFS getter.
        
    Maps variables to the common names used in shared functions.
    By default adds mapping {"lon": "longitude", "lat": "latitude"}
    """

    def __init__(self, getter, alternate_keys: Mapping[str, str] = None):
        self._getter = getter
        self._alternate_keys = alternate_keys or {
            "lon": "longitude",
            "lat": "latitude",
            "physics_precip": PHYSICS_PRECIP_RATE,
        }

    def __getitem__(self, key: str) -> xr.DataArray:
        if key == "time":
            time = self._getter.get_state(["time"])["time"]
            return xr.DataArray(time, name="time")
        elif key == "latent_heat_flux":
            return self._getter.get_diagnostic_by_name("lhtfl").data_array
        elif key == "total_water":
            return self._total_water()
        else:
            if key in self._alternate_keys:
                key = self._alternate_keys[key]
            try:
                return self._getter.get_state([key])[key].data_array
            except pace.util.InvalidQuantityError as e:
                raise KeyError(e)

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
        tracer_names = set(v for v in self._getter.get_tracer_metadata())
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
    
    This encapsulates from the details of Quantity
    
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
        state_time = self._getter.get_state(["time"])["time"]
        return round_time(cftime.DatetimeJulian(*state_time.timetuple()))

    def __getitem__(self, key: Hashable) -> xr.DataArray:
        return self._mapper[key]

    def __setitem__(self, key: str, value: xr.DataArray):
        state_update = _cast_single_to_double({key: value})
        try:
            self._getter.set_state_mass_conserving(
                _data_arrays_to_quantities(state_update)
            )
        except ValueError as e:
            raise KeyError(e)

    def keys(self):
        return self._mapper.keys()

    def update_mass_conserving(
        self, items: State,
    ):
        """Update state from another mapping

        This may be faster than setting each item individually. Same as dict.update.
        
        All states except for pressure thicknesses are set in a mass-conserving fashion.
        """
        items_with_attrs = _cast_single_to_double(self._assign_attrs_from_mapper(items))

        if DELP in items_with_attrs:
            self._getter.set_state(
                _data_arrays_to_quantities({DELP: items_with_attrs[DELP]})
            )

        not_pressure = dissoc(items_with_attrs, DELP)
        try:
            self._getter.set_state_mass_conserving(
                _data_arrays_to_quantities(not_pressure)
            )
        except ValueError as e:
            raise KeyError(e)

    def _assign_attrs_from_mapper(self, dst: State) -> State:
        updated = {}
        for name in dst:
            updated[name] = dst[name].assign_attrs(self._mapper[name].attrs)
        return updated

    def __delitem__(self, key: str):
        raise NotImplementedError()

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(self.keys())


class MergedState(DerivedFV3State):
    """Merge a State object with another mutable mapping

    Commonly used to e.g. blend python and fv3gfs-fortran state into a single
    object.
    
    Attributes:
        left: a derived fv3 state object represetning e.g. Fortran state.
        right: a mutable mapping representing python based state.
            Used for all keys not in ``left``.

    """

    def __init__(self, left: DerivedFV3State, right: State):
        self.left = left
        self.right = right

    def update_mass_conserving(
        self, items: State,
    ):
        self.left.update_mass_conserving(items)

    @property
    def time(self) -> cftime.DatetimeJulian:
        return self.left.time

    def __setitem__(self, key: str, value: xr.DataArray):
        try:
            self.left[key] = value
        except KeyError:
            self.right[key] = value

    def __getitem__(self, key: Hashable) -> xr.DataArray:
        try:
            return self.left[key]
        except KeyError:
            return self.right[key]

    def keys(self) -> Set[str]:
        all_keys = set(self.left.keys()) | set(self.right.keys())
        return set(map(str, all_keys))

    def __delitem__(self, key: str):
        try:
            del self.left[key]
        except (KeyError, NotImplementedError):
            del self.right[key]


def _cast_single_to_double(state: State) -> State:
    # wrapper state variables must be in double precision
    cast_state = {}
    for name in state:
        if state[name].values.dtype == np.float32:
            cast_state[name] = (
                state[name]
                .astype(np.float64, casting="same_kind")
                .assign_attrs(state[name].attrs)
            )
        else:
            cast_state[name] = state[name]
    return cast_state


def _data_arrays_to_quantities(state: State) -> Mapping[Hashable, pace.util.Quantity]:
    return {
        key: pace.util.Quantity.from_data_array(value) for key, value in state.items()
    }
