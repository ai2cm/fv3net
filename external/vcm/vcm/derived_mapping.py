from typing import Mapping, Hashable, Callable, Sequence
import xarray as xr

import vcm


class DerivedMapping:
    """A uniform mapping-like interface for both existing and derived variables.
    
    Allows register and computing derived variables transparently in either
    the FV3GFS state or a saved dataset.

    """

    _VARIABLES: Mapping[Hashable, Callable[..., xr.DataArray]] = {}

    def __init__(self, mapper: Mapping[str, xr.DataArray]):
        self._mapper = mapper

    @classmethod
    def register(cls, name: str):
        """Register a function as a derived variable

        Args:
            name: the name the derived variable will be available under
        """

        def decorator(func):
            cls._VARIABLES[name] = func
            return func

        return decorator

    def __getitem__(self, key: str) -> xr.DataArray:
        if key in self._VARIABLES:
            return self._VARIABLES[key](self)
        else:
            return self._mapper[key]

    def _data_arrays(self, keys: Sequence[str]):
        return {key: self[key] for key in keys}

    def dataset(self, keys: Sequence[str]) -> xr.Dataset:
        return xr.Dataset(self._data_arrays(keys))


@DerivedMapping.register("cos_zenith_angle")
def cos_zenith_angle(self):
    return xr.apply_ufunc(
        lambda time, lon, lat: vcm.cos_zenith_angle(time, lon, lat),
        self["time"],
        self["lon"],
        self["lat"],
        dask="allowed",
    )


@DerivedMapping.register("evaporation")
def evaporation(self):
    lhf = self["latent_heat_flux"]
    return vcm.thermo.latent_heat_flux_to_evaporation(lhf)
