import numpy as np
from typing import Mapping, Hashable, Callable, Iterable
import xarray as xr

import vcm


class DerivedMapping:
    """A uniform mapping-like interface for both existing and derived variables.
    
    Allows register and computing derived variables transparently in either
    the FV3GFS state or a saved dataset.

    """

    VARIABLES: Mapping[Hashable, Callable[..., xr.DataArray]] = {}
    REQUIRED_INPUTS: Mapping[Hashable, Iterable[Hashable]] = {}

    def __init__(self, mapper: Mapping[Hashable, xr.DataArray]):
        self._mapper = mapper

    @classmethod
    def register(cls, name: Hashable, required_inputs: Iterable[Hashable] = None):
        """Register a function as a derived variable.

        Args:
            name: the name the derived variable will be available under
            required_inputs: Optional arg to list the potential
                required inputs needed to derive said variable. Even if the
                requirements are not well-defined, they should still be listed.
                (e.g. dQu only needs dQxwind, dQywind if dQu is not in the data)
                This is because the usage of this registry is for when
                an output is explicitly requested as a derived output variable and thus
                it is assumed the variable does not already exist and needs to
                be derived.
        """

        def decorator(func):
            cls.VARIABLES[name] = func
            if required_inputs:
                cls.REQUIRED_INPUTS[name] = required_inputs
            return func

        return decorator

    def __getitem__(self, key: Hashable) -> xr.DataArray:
        if key in self.VARIABLES:
            return self.VARIABLES[key](self)
        else:
            return self._mapper[key]

    def keys(self):
        return set(self._mapper) | set(self.VARIABLES)

    def _data_arrays(self, keys: Iterable[Hashable]):
        return {key: self[key] for key in keys}

    def dataset(self, keys: Iterable[Hashable]) -> xr.Dataset:
        return xr.Dataset(self._data_arrays(keys))

    @classmethod
    def find_all_required_inputs(
        cls, derived_variables: Iterable[Hashable]
    ) -> Iterable[Hashable]:
        # Helper function to find full list of required inputs for a given list
        # of derived variables. Recurses because some required inputs have their
        # own required inputs (e.g. pQ's)
        def _recurse_find_deps(vars, deps):
            vars_with_deps = [var for var in vars if var in cls.REQUIRED_INPUTS]
            if len(vars_with_deps) == 0:
                return
            else:
                for var in vars_with_deps:
                    new_deps = cls.REQUIRED_INPUTS[var]
                    deps += new_deps
                _recurse_find_deps(new_deps, deps)

        deps: Iterable[Hashable] = []
        _recurse_find_deps(derived_variables, deps)
        return deps


@DerivedMapping.register("cos_zenith_angle", required_inputs=["time", "lon", "lat"])
def cos_zenith_angle(self):
    return vcm.cos_zenith_angle(self["time"], self["lon"], self["lat"])


@DerivedMapping.register("evaporation", required_inputs=["latent_heat_flux"])
def evaporation(self):
    lhf = self["latent_heat_flux"]
    return vcm.thermo.latent_heat_flux_to_evaporation(lhf)


def _rotate(self: DerivedMapping, x, y):
    wind_rotation_matrix = self.dataset(
        [
            "eastward_wind_u_coeff",
            "eastward_wind_v_coeff",
            "northward_wind_u_coeff",
            "northward_wind_v_coeff",
        ]
    )
    return vcm.cubedsphere.center_and_rotate_xy_winds(
        wind_rotation_matrix, self[x], self[y]
    )


@DerivedMapping.register("dQu", required_inputs=["dQxwind", "dQywind"])
def dQu(self):
    try:
        return self._mapper["dQu"]
    except (KeyError):
        return _rotate(self, "dQxwind", "dQywind")[0]


@DerivedMapping.register("dQv", required_inputs=["dQxwind", "dQywind"])
def dQv(self):
    try:
        return self._mapper["dQv"]
    except (KeyError):
        return _rotate(self, "dQxwind", "dQywind")[1]


@DerivedMapping.register("eastward_wind")
def eastward_wind(self):
    try:
        return self._mapper["eastward_wind"]
    except (KeyError):
        return _rotate(self, "x_wind", "y_wind")[0]


@DerivedMapping.register("northward_wind")
def northward_wind(self):
    try:
        return self._mapper["northward_wind"]
    except (KeyError):
        return _rotate(self, "x_wind", "y_wind")[1]


@DerivedMapping.register(
    "dQu_parallel_to_eastward_wind", required_inputs=["eastward_wind", "dQu"]
)
def dQu_parallel_to_eastward_wind_direction(self):
    sign = np.sign(self["eastward_wind"] / self["dQu"])
    return sign * abs(self["dQu"])


@DerivedMapping.register(
    "dQv_parallel_to_northward_wind", required_inputs=["northward_wind", "dQv"]
)
def dQv_parallel_to_northward_wind_direction(self):
    sign = np.sign(self["northward_wind"] / self["dQv"])
    return sign * abs(self["dQv"])


@DerivedMapping.register(
    "horizontal_wind_tendency_parallel_to_horizontal_wind",
    required_inputs=["eastward_wind", "dQu", "northward_wind", "dQv"],
)
def horizontal_wind_tendency_parallel_to_horizontal_wind(self):
    tendency_projection_onto_wind = (
        self["eastward_wind"] * self["dQu"] + self["northward_wind"] * self["dQv"]
    ) / np.linalg.norm((self["eastward_wind"], self["northward_wind"]))
    return tendency_projection_onto_wind


@DerivedMapping.register(
    "net_shortwave_sfc_flux_derived",
    required_inputs=["surface_diffused_shortwave_albedo"],
)
def net_shortwave_sfc_flux_derived(self):
    # Positive = downward direction
    albedo = self["surface_diffused_shortwave_albedo"]
    downward_sfc_shortwave_flux = self[
        "override_for_time_adjusted_total_sky_downward_shortwave_flux_at_surface"
    ]
    return (1 - albedo) * downward_sfc_shortwave_flux


@DerivedMapping.register(
    "is_land", required_inputs=["land_sea_mask"],
)
def is_land(self):
    # one hot encoding for land / (sea or sea ice) surface
    return xr.where(vcm.xarray_utils.isclose(self["land_sea_mask"], 1), 1.0, 0.0)


@DerivedMapping.register(
    "is_sea", required_inputs=["land_sea_mask"],
)
def is_sea(self):
    # one hot encoding for sea surface
    return xr.where(vcm.xarray_utils.isclose(self["land_sea_mask"], 0), 1.0, 0.0)


@DerivedMapping.register(
    "is_sea_ice", required_inputs=["land_sea_mask"],
)
def is_sea_ice(self):
    # one hot encoding for sea ice surface
    return xr.where(vcm.xarray_utils.isclose(self["land_sea_mask"], 2), 1.0, 0.0)


@DerivedMapping.register("Q1", required_inputs=["pQ1"])
def Q1(self):
    try:
        return self._mapper["Q1"]
    except KeyError:
        if "dQ1" in self.keys():
            return self["dQ1"] + self["pQ1"]
        else:
            return self["pQ1"]


@DerivedMapping.register("Q2", required_inputs=["pQ2"])
def Q2(self):
    try:
        return self._mapper["Q2"]
    except KeyError:
        if "dQ2" in self.keys():
            return self["dQ2"] + self["pQ2"]
        else:
            return self["pQ2"]


@DerivedMapping.register(
    "pQ1", required_inputs=["pressure_thickness_of_atmospheric_layer"]
)
def pQ1(self):
    try:
        return self._mapper["pQ1"]
    except KeyError:
        return xr.zeros_like(self["pressure_thickness_of_atmospheric_layer"])


@DerivedMapping.register(
    "pQ2", required_inputs=["pressure_thickness_of_atmospheric_layer"]
)
def pQ2(self):
    try:
        return self._mapper["pQ2"]
    except KeyError:
        return xr.zeros_like(self["pressure_thickness_of_atmospheric_layer"])
