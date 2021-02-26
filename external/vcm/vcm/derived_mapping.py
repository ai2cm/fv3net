import numpy as np
import xarray as xr
from typing import Mapping, Hashable, Callable, Sequence

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

    def keys(self):
        return set(self._mapper) | set(self._VARIABLES)

    def _data_arrays(self, keys: Sequence[str]):
        return {key: self[key] for key in keys}

    def dataset(self, keys: Sequence[str]) -> xr.Dataset:
        return xr.Dataset(self._data_arrays(keys))


@DerivedMapping.register("cos_zenith_angle")
def cos_zenith_angle(self):
    return vcm.cos_zenith_angle(self["time"], self["lon"], self["lat"])


@DerivedMapping.register("evaporation")
def evaporation(self):
    lhf = self["latent_heat_flux"]
    return vcm.thermo.latent_heat_flux_to_evaporation(lhf)


@DerivedMapping.register("dQu")
def dQu(self):
    # try/except is a placeholder for a future PR to add keys to
    # DerivedMapping so that the class will return this key if it already
    # exists, and derive it if not. This is currently blocked because the
    # FV3 wrapper needs a function to get available field names.
    try:
        return self._mapper["dQu"]
    except (KeyError):
        wind_rotation_matrix = self.dataset(
            [
                "eastward_wind_u_coeff",
                "eastward_wind_v_coeff",
                "northward_wind_u_coeff",
                "northward_wind_v_coeff",
            ]
        )
        return vcm.cubedsphere.center_and_rotate_xy_winds(
            wind_rotation_matrix, self["dQxwind"], self["dQywind"]
        )[0]


@DerivedMapping.register("dQv")
def dQv(self):
    # try/except is a placeholder for a future PR to add keys to
    # DerivedMapping so that the class will return this key if it already
    # exists, and derive it if not. This is currently blocked because the
    # FV3 wrapper needs a function to get available field names.
    try:
        return self._mapper["dQv"]
    except (KeyError):
        wind_rotation_matrix = self.dataset(
            [
                "eastward_wind_u_coeff",
                "eastward_wind_v_coeff",
                "northward_wind_u_coeff",
                "northward_wind_v_coeff",
            ]
        )
        return vcm.cubedsphere.center_and_rotate_xy_winds(
            wind_rotation_matrix, self["dQxwind"], self["dQywind"]
        )[1]


@DerivedMapping.register("dQu_parallel_to_eastward_wind")
def dQu_parallel_to_eastward_wind_direction(self):
    sign = np.sign(self["eastward_wind"] / self["dQu"])
    return sign * abs(self["dQu"])


@DerivedMapping.register("dQv_parallel_to_northward_wind")
def dQv_parallel_to_northward_wind_direction(self):
    sign = np.sign(self["northward_wind"] / self["dQv"])
    return sign * abs(self["dQv"])


@DerivedMapping.register("horizontal_wind_tendency_parallel_to_horizontal_wind")
def horizontal_wind_tendency_parallel_to_horizontal_wind(self):
    tendency_projection_onto_wind = (
        self["eastward_wind"] * self["dQu"] + self["northward_wind"] * self["dQv"]
    ) / np.linalg.norm((self["eastward_wind"], self["northward_wind"]))
    return tendency_projection_onto_wind


def _get_datetime_attr(self, attr_name) -> xr.DataArray:
    times = self["time"].values
    time_attr = [getattr(t, attr_name) for t in times]
    return xr.DataArray(data=time_attr, dims=["time"])


@DerivedMapping.register("cos_day")
def cos_day(self):
    return np.cos(2 * np.pi * _get_datetime_attr(self, "dayofyr") / 366)


@DerivedMapping.register("sin_day")
def sin_day(self):
    return np.sin(2 * np.pi * _get_datetime_attr(self, "dayofyr") / 366)


@DerivedMapping.register("cos_month")
def cos_month(self):
    return np.cos(2 * np.pi * _get_datetime_attr(self, "month") / 12)


@DerivedMapping.register("sin_month")
def sin_month(self):
    return np.sin(2 * np.pi * _get_datetime_attr(self, "month") / 12)


@DerivedMapping.register("cos_lon")
def cos_lon(self):
    return np.cos(self["longitude"])


@DerivedMapping.register("sin_lon")
def sin_lon(self):
    return np.sin(self["longitude"])
