import dataclasses
from toolz.functoolz import curry
from typing import Callable, Literal, MutableMapping, Sequence, Set
import xarray as xr
import vcm


@dataclasses.dataclass
class DataTransformRegistryEntry:
    func: Callable[..., xr.Dataset]
    inputs: Sequence[str]
    outputs: Sequence[str]


DATA_TRANSFORM_REGISTRY: MutableMapping[str, DataTransformRegistryEntry] = {}

TransformName = Literal[
    "Q1_from_Qm_Q2",
    "Qm_from_Q1_Q2",
    "Q1_from_Qm_Q2_temperature_dependent",
    "Qm_from_Q1_Q2_temperature_dependent",
    "Q1_from_dQ1_pQ1",
    "Q2_from_dQ2_pQ2",
    "Qm_flux_from_Qm_tendency",
    "Q2_flux_from_Q2_tendency",
    "Qm_tendency_from_Qm_flux",
    "Q2_tendency_from_Q2_flux",
]


@curry
def register(
    inputs: Sequence[str], outputs: Sequence[str], func: Callable[..., xr.Dataset],
):
    name = func.__name__
    if name in DATA_TRANSFORM_REGISTRY:
        raise ValueError(f"Function {name} has already been added to registry.")
    DATA_TRANSFORM_REGISTRY[name] = DataTransformRegistryEntry(
        func=func, inputs=inputs, outputs=outputs
    )
    return func


@register(["Q1", "Q2"], ["Qm"])
def Qm_from_Q1_Q2(ds):
    ds["Qm"] = vcm.moist_static_energy_tendency(ds["Q1"], ds["Q2"])
    return ds


@register(["Qm", "Q2"], ["Q1"])
def Q1_from_Qm_Q2(ds):
    ds["Q1"] = vcm.temperature_tendency(ds["Qm"], ds["Q2"])
    return ds


@register(["Q1", "Q2", "air_temperature"], ["Qm"])
def Qm_from_Q1_Q2_temperature_dependent(ds):
    ds["Qm"] = vcm.moist_static_energy_tendency(
        ds["Q1"], ds["Q2"], temperature=ds["air_temperature"]
    )
    return ds


@register(["Qm", "Q2", "air_temperature"], ["Q1"])
def Q1_from_Qm_Q2_temperature_dependent(ds):
    ds["Q1"] = vcm.temperature_tendency(
        ds["Qm"], ds["Q2"], temperature=ds["air_temperature"]
    )
    return ds


@register(["dQ1", "pQ1"], ["Q1"])
def Q1_from_dQ1_pQ1(ds):
    ds["Q1"] = ds["dQ1"] + ds["pQ1"]
    return ds


@register(["dQ2", "pQ2"], ["Q2"])
def Q2_from_dQ2_pQ2(ds):
    ds["Q2"] = ds["dQ2"] + ds["pQ2"]
    return ds


DELP = "pressure_thickness_of_atmospheric_layer"
DLW_SFC = "total_sky_downward_longwave_flux_at_surface"
DSW_SFC = "total_sky_downward_shortwave_flux_at_surface"
DSW_TOA = "total_sky_downward_shortwave_flux_at_top_of_atmosphere"
ULW_SFC = "total_sky_upward_longwave_flux_at_surface"
ULW_TOA = "total_sky_upward_longwave_flux_at_top_of_atmosphere"
USW_SFC = "total_sky_upward_shortwave_flux_at_surface"
USW_TOA = "total_sky_upward_shortwave_flux_at_top_of_atmosphere"
COL_T_NUDGE = "storage_of_internal_energy_path_due_to_fine_res_temperature_nudging"
LHF = "latent_heat_flux"
SHF = "sensible_heat_flux"
PRECIP = "surface_precipitation_rate"


@register(
    [
        "Qm",
        DELP,
        DLW_SFC,
        DSW_SFC,
        DSW_TOA,
        ULW_SFC,
        ULW_TOA,
        USW_SFC,
        USW_TOA,
        LHF,
        SHF,
        COL_T_NUDGE,
    ],
    ["Qm_flux", "implied_downward_radiative_flux_at_surface"],
)
def Qm_flux_from_Qm_tendency(
    ds, rectify_downward_radiative_flux=True, include_temperature_nudging=True,
):
    Qm_flux = -vcm.mass_cumsum(ds["Qm"], ds[DELP], dim="z")
    toa_rad_flux = ds[DSW_TOA] - ds[USW_TOA] - ds[ULW_TOA]
    upward_surface_mse_flux = ds[LHF] + ds[SHF] + ds[USW_SFC] + ds[ULW_SFC]

    # add TOA boundary condition
    Qm_flux = Qm_flux.pad(z=(1, 0), constant_values=0.0)
    Qm_flux += toa_rad_flux
    if include_temperature_nudging:
        Qm_flux += ds[COL_T_NUDGE]

    # compute downward flux at surface
    downward_sfc_Qm_flux = Qm_flux.isel(z=-1) + upward_surface_mse_flux
    if rectify_downward_radiative_flux:
        downward_sfc_Qm_flux = downward_sfc_Qm_flux.where(downward_sfc_Qm_flux >= 0, 0)

    # remove bottom flux level from net fluxes
    Qm_flux = Qm_flux.isel(z=slice(None, -1))

    ds["Qm_flux"] = Qm_flux.chunk({"z": Qm_flux.sizes["z"]})
    ds["implied_downward_radiative_flux_at_surface"] = downward_sfc_Qm_flux
    return ds


@register(
    ["Q2", DELP, LHF, PRECIP], ["Q2_flux", "implied_surface_precipitation_rate"],
)
def Q2_flux_from_Q2_tendency(
    ds, rectify_surface_precipitation_rate=True,
):
    Q2_flux = -vcm.mass_cumsum(ds["Q2"], ds[DELP], dim="z")
    evaporation = vcm.latent_heat_flux_to_evaporation(ds[LHF])

    # pad at top for TOA flux (boundary condition for Q2_flux=0)
    Q2_flux = Q2_flux.pad(z=(1, 0), constant_values=0.0)

    # compute downward flux at surface
    downward_sfc_Q2_flux = Q2_flux.isel(z=-1) + evaporation
    if rectify_surface_precipitation_rate:
        downward_sfc_Q2_flux = downward_sfc_Q2_flux.where(downward_sfc_Q2_flux >= 0, 0)

    # remove bottom flux level from net fluxes
    Q2_flux = Q2_flux.isel(z=slice(None, -1))

    ds["Q2_flux"] = Q2_flux.chunk({"z": Q2_flux.sizes["z"]})
    ds["implied_surface_precipitation_rate"] = downward_sfc_Q2_flux
    return ds


@register(
    [
        "Qm_flux",
        "implied_downward_radiative_flux_at_surface",
        DELP,
        ULW_SFC,
        USW_SFC,
        LHF,
        SHF,
    ],
    ["Qm"],
)
def Qm_tendency_from_Qm_flux(ds):
    upward_surface_mse_flux = ds[LHF] + ds[SHF] + ds[USW_SFC] + ds[ULW_SFC]
    net_surface_mse_flux = (
        ds["implied_downward_radiative_flux_at_surface"] - upward_surface_mse_flux
    )
    Qm_flux = xr.concat([ds["Qm_flux"], net_surface_mse_flux], dim="z")
    Qm_flux = Qm_flux.chunk({"z": Qm_flux.sizes["z"]})
    ds["Qm"] = -vcm.mass_divergence(
        Qm_flux, ds[DELP], dim_center="z", dim_interface="z"
    )
    return ds


@register(
    ["Q2_flux", "implied_surface_precipitation_rate", DELP, LHF], ["Q2"],
)
def Q2_tendency_from_Q2_flux(ds):
    evaporation = vcm.latent_heat_flux_to_evaporation(ds[LHF])
    net_surface_q2_flux = ds["implied_surface_precipitation_rate"] - evaporation
    Q2_flux = xr.concat([ds["Q2_flux"], net_surface_q2_flux], dim="z")
    Q2_flux = Q2_flux.chunk({"z": Q2_flux.sizes["z"]})
    ds["Q2"] = -vcm.mass_divergence(
        Q2_flux, ds[DELP], dim_center="z", dim_interface="z"
    )
    return ds


@dataclasses.dataclass
class DataTransform:
    name: TransformName
    kwargs: dict = dataclasses.field(default_factory=dict)

    def apply(self, ds: xr.Dataset) -> xr.Dataset:
        func = DATA_TRANSFORM_REGISTRY[self.name].func
        ds = func(ds, **self.kwargs)
        return ds

    @property
    def input_variables(self) -> Sequence[str]:
        return DATA_TRANSFORM_REGISTRY[self.name].inputs

    @property
    def output_variables(self) -> Sequence[str]:
        return DATA_TRANSFORM_REGISTRY[self.name].outputs


@dataclasses.dataclass
class ChainedDataTransform:
    transforms: Sequence[DataTransform]

    def apply(self, ds: xr.Dataset) -> xr.Dataset:
        for transform in self.transforms:
            ds = transform.apply(ds)
        return ds

    @property
    def input_variables(self) -> Sequence[str]:
        inputs: Set[str] = set()
        for transform in self.transforms[::-1]:
            inputs.update(transform.input_variables)
            for output in transform.output_variables:
                inputs.discard(output)
        return sorted(list(inputs))

    @property
    def output_variables(self) -> Sequence[str]:
        outputs: Set[str] = set()
        for transform in self.transforms:
            outputs.update(transform.output_variables)
        return sorted(list(outputs))
