import numpy as np
from typing import Mapping, Hashable, Callable, Iterable, MutableMapping
import xarray as xr

import vcm


class DerivedMapping(Mapping):
    """A uniform mapping-like interface for both existing and derived variables.
    
    Allows register and computing derived variables transparently in either
    the FV3GFS state or a saved dataset.

    """

    VARIABLES: MutableMapping[Hashable, Callable[..., xr.DataArray]] = {}
    REQUIRED_INPUTS: MutableMapping[Hashable, Iterable[Hashable]] = {}
    USE_NONDERIVED_IF_EXISTS: Iterable[Hashable] = []

    def __init__(self, mapper: Mapping[Hashable, xr.DataArray]):
        self._mapper = mapper

    @classmethod
    def register(
        cls,
        name: Hashable,
        required_inputs: Iterable[Hashable] = None,
        use_nonderived_if_exists: bool = False,
    ):
        """Register a function as a derived variable.

        Args:
            name: the name the derived variable will be available under
            required_inputs:
                List of the the required inputs needed to derive said
                variable. Even if the requirements are not well-defined, they should
                still be listed. (e.g. dQu only needs dQxwind, dQywind if dQu is not
                in the data). This is because the usage of this registry is for when
                an output is explicitly requested as a derived output variable and thus
                it is assumed the variable does not already exist and needs to
                be derived.

                Only the direct dependencies need to be listed here. e.g. if derived
                variable "a" requires "b", and "b" requires "c", the required_inputs
                for "a" should just be ["b"].
            use_nonderived_if_exists:
                Some variables may exist in the data already. If this flag is True,
                first check if they exist and return existing values if so. If the
                variable in not in the underlying data, the derived mapping will
                calculate it and return those values.
        """

        def decorator(func):
            cls.VARIABLES[name] = func
            if required_inputs:
                cls.REQUIRED_INPUTS[name] = required_inputs
            if use_nonderived_if_exists is True:
                cls.USE_NONDERIVED_IF_EXISTS.append(name)
            return func

        return decorator

    def __getitem__(self, key: Hashable) -> xr.DataArray:
        if key in self.VARIABLES:
            if key in self.USE_NONDERIVED_IF_EXISTS:
                try:
                    return self._mapper[key]
                except (KeyError):
                    return self.VARIABLES[key](self)
            else:
                return self.VARIABLES[key](self)
        else:
            return self._mapper[key]

    def keys(self):
        return set(self._mapper) | set(self.VARIABLES)

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(self.keys())

    def _data_arrays(self, keys: Iterable[Hashable]):
        return {key: self[key] for key in keys}

    def dataset(self, keys: Iterable[Hashable]) -> xr.Dataset:
        return xr.Dataset(self._data_arrays(keys))

    @classmethod
    def find_all_required_inputs(
        cls, derived_variables: Iterable[Hashable]
    ) -> Iterable[Hashable]:
        # Helper function to find full list of required (non-derived) inputs for
        # a given list of derived variables. Recurses because some required inputs
        # have their own required inputs (e.g. pQ's). Excludes intermediate required
        # inputs that are themselves derived.

        def _recurse_find_deps(vars, deps):
            vars_with_deps = [var for var in vars if var in cls.REQUIRED_INPUTS]
            if len(vars_with_deps) == 0:
                return
            else:
                new_deps = []
                for var in vars_with_deps:
                    new_deps += cls.REQUIRED_INPUTS[var]
                deps += new_deps
                _recurse_find_deps(new_deps, deps)

        deps: Iterable[Hashable] = []
        _recurse_find_deps(derived_variables, deps)
        # omit intermediate inputs unless they are in list of variables
        # to use from existing data if present
        nonderived_deps = list(set([dep for dep in deps if dep not in cls.VARIABLES]))
        maybe_nonderived_deps = list(
            set([dep for dep in deps if dep in cls.USE_NONDERIVED_IF_EXISTS])
        )
        return nonderived_deps + maybe_nonderived_deps


@DerivedMapping.register("cos_zenith_angle", required_inputs=["time", "lon", "lat"])
def cos_zenith_angle(self):
    return vcm.cos_zenith_angle(self["time"], self["lon"], self["lat"])


@DerivedMapping.register("evaporation", required_inputs=["latent_heat_flux"])
def evaporation(self):
    lhf = self["latent_heat_flux"]
    return vcm.latent_heat_flux_to_evaporation(lhf)


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


@DerivedMapping.register(
    "dQu", required_inputs=["dQxwind", "dQywind"], use_nonderived_if_exists=True
)
def dQu(self):
    return _rotate(self, "dQxwind", "dQywind")[0]


@DerivedMapping.register(
    "dQv", required_inputs=["dQxwind", "dQywind"], use_nonderived_if_exists=True
)
def dQv(self):
    return _rotate(self, "dQxwind", "dQywind")[1]


@DerivedMapping.register("eastward_wind", use_nonderived_if_exists=True)
def eastward_wind(self):
    return _rotate(self, "x_wind", "y_wind")[0]


@DerivedMapping.register("northward_wind", use_nonderived_if_exists=True)
def northward_wind(self):
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


def _net_sfc_shortwave_flux_via_albedo(downward_sfc_shortwave_flux, albedo):
    return (1 - albedo) * downward_sfc_shortwave_flux


@DerivedMapping.register(
    "net_shortwave_sfc_flux_derived",
    required_inputs=[
        "surface_diffused_shortwave_albedo",
        "override_for_time_adjusted_total_sky_downward_shortwave_flux_at_surface",
    ],
)
def net_shortwave_sfc_flux_derived(self):
    # Positive = downward direction
    albedo = self["surface_diffused_shortwave_albedo"]
    downward_sfc_shortwave_flux = self[
        "override_for_time_adjusted_total_sky_downward_shortwave_flux_at_surface"
    ]
    return _net_sfc_shortwave_flux_via_albedo(downward_sfc_shortwave_flux, albedo)


@DerivedMapping.register(
    "downward_shortwave_sfc_flux_via_transmissivity",
    required_inputs=[
        "total_sky_downward_shortwave_flux_at_top_of_atmosphere",
        "shortwave_transmissivity_of_atmospheric_column",
    ],
)
def downward_shortwave_sfc_flux_via_transmissivity(self):
    toa_flux = self["total_sky_downward_shortwave_flux_at_top_of_atmosphere"]
    transmissivity = self["shortwave_transmissivity_of_atmospheric_column"]
    return transmissivity * toa_flux


@DerivedMapping.register(
    "net_shortwave_sfc_flux_via_transmissivity",
    required_inputs=[
        "surface_diffused_shortwave_albedo",
        "total_sky_downward_shortwave_flux_at_top_of_atmosphere",
        "downward_shortwave_sfc_flux_via_transmissivity",
    ],
)
def net_shortwave_sfc_flux_via_transmissivity(self):
    downward_sfc_shortwave_flux = self["downward_shortwave_sfc_flux_via_transmissivity"]
    albedo = self["surface_diffused_shortwave_albedo"]
    return _net_sfc_shortwave_flux_via_albedo(downward_sfc_shortwave_flux, albedo)


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


@DerivedMapping.register("Q1", required_inputs=["pQ1"], use_nonderived_if_exists=True)
def Q1(self):
    if "dQ1" in self.keys():
        return self["dQ1"] + self["pQ1"]
    else:
        return self["pQ1"]


@DerivedMapping.register("Q2", required_inputs=["pQ2"], use_nonderived_if_exists=True)
def Q2(self):
    if "dQ2" in self.keys():
        return self["dQ2"] + self["pQ2"]
    else:
        return self["pQ2"]


@DerivedMapping.register(
    "pQ1",
    required_inputs=["pressure_thickness_of_atmospheric_layer"],
    use_nonderived_if_exists=True,
)
def pQ1(self):
    return xr.zeros_like(self["pressure_thickness_of_atmospheric_layer"])


@DerivedMapping.register(
    "pQ2",
    required_inputs=["pressure_thickness_of_atmospheric_layer"],
    use_nonderived_if_exists=True,
)
def pQ2(self):
    return xr.zeros_like(self["pressure_thickness_of_atmospheric_layer"])


@DerivedMapping.register("internal_energy", required_inputs=["air_temperature"])
def internal_energy(self):
    return vcm.internal_energy(self._mapper["air_temperature"])


@DerivedMapping.register(
    "column_integrated_dQ1",
    required_inputs=["dQ1", "pressure_thickness_of_atmospheric_layer"],
)
def column_integrated_dQ1(self):
    return vcm.column_integrated_heating_from_isochoric_transition(
        self._mapper["dQ1"], self._mapper["pressure_thickness_of_atmospheric_layer"]
    )


@DerivedMapping.register(
    "column_integrated_dQ2",
    required_inputs=["dQ2", "pressure_thickness_of_atmospheric_layer"],
)
def column_integrated_dQ2(self):
    da = -vcm.minus_column_integrated_moistening(
        self._mapper["dQ2"], self._mapper["pressure_thickness_of_atmospheric_layer"]
    )
    return da.assign_attrs(
        {"long_name": "column integrated moistening", "units": "mm/day"}
    )


@DerivedMapping.register(
    "column_integrated_Q1",
    required_inputs=["Q1", "pressure_thickness_of_atmospheric_layer"],
)
def column_integrated_Q1(self):
    return vcm.column_integrated_heating_from_isochoric_transition(
        self._mapper["Q1"], self._mapper["pressure_thickness_of_atmospheric_layer"]
    )


@DerivedMapping.register(
    "column_integrated_Q2",
    required_inputs=["Q2", "pressure_thickness_of_atmospheric_layer"],
)
def column_integrated_Q2(self):
    da = -vcm.minus_column_integrated_moistening(
        self._mapper["Q2"], self._mapper["pressure_thickness_of_atmospheric_layer"]
    )
    return da.assign_attrs(
        {"long_name": "column integrated moistening", "units": "mm/day"}
    )


@DerivedMapping.register(
    "water_vapor_path",
    required_inputs=["specific_humidity", "pressure_thickness_of_atmospheric_layer"],
    use_nonderived_if_exists=True,
)
def water_vapor_path(self):
    da = vcm.mass_integrate(
        self._mapper["specific_humidity"],
        self._mapper["pressure_thickness_of_atmospheric_layer"],
        dim="z",
    )
    return da.assign_attrs(
        {"long_name": "column integrated water vapor", "units": "mm"}
    )
