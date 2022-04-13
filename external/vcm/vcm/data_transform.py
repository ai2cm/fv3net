import dataclasses
from toolz.functoolz import curry
from typing import Callable, Literal, MutableMapping, Sequence, Set
import xarray as xr
import vcm
from .calc.flux_form import (
    _tendency_to_flux,
    _flux_to_tendency,
    _tendency_to_implied_surface_downward_flux,
)


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
    "implied_surface_precipitation_rate",
    "implied_downward_radiative_flux_at_surface",
]


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
    """See https://github.com/ai2cm/explore/blob/master/oliwm/
    2021-12-13-fine-res-in-flux-form/2021-12-13-fine-res-in-flux-form-proposal-v2.ipynb
    for derivation of transform."""
    toa_net_flux = ds[DSW_TOA] - ds[USW_TOA] - ds[ULW_TOA]
    if include_temperature_nudging:
        toa_net_flux += ds[COL_T_NUDGE]
    surface_upward_flux = ds[LHF] + ds[SHF] + ds[USW_SFC] + ds[ULW_SFC]
    net_flux, surface_downward_flux = _tendency_to_flux(
        ds["Qm"],
        toa_net_flux,
        surface_upward_flux,
        ds[DELP],
        dim="z",
        rectify=rectify_downward_radiative_flux,
    )
    surface_downward_flux = surface_downward_flux.assign_attrs(
        units="W/m**2",
        long_name="Implied downward radiative flux from <Qm> budget closure",
    )
    ds["Qm_flux"] = net_flux.assign_attrs(units="W/m**2", long_name="Net flux of MSE")
    ds["implied_downward_radiative_flux_at_surface"] = surface_downward_flux
    return ds


@register(
    ["Q2", DELP, LHF], ["Q2_flux", "implied_surface_precipitation_rate"],
)
def Q2_flux_from_Q2_tendency(
    ds, rectify_surface_precipitation_rate=True,
):
    """See https://github.com/ai2cm/explore/blob/master/oliwm/
    2021-12-13-fine-res-in-flux-form/2021-12-13-fine-res-in-flux-form-proposal-v2.ipynb
    for derivation of transform."""
    toa_net_flux = xr.zeros_like(ds[LHF])
    surface_upward_flux = vcm.latent_heat_flux_to_evaporation(ds[LHF])
    net_flux, surface_downward_flux = _tendency_to_flux(
        ds["Q2"],
        toa_net_flux,
        surface_upward_flux,
        ds[DELP],
        dim="z",
        rectify=rectify_surface_precipitation_rate,
    )
    surface_downward_flux = surface_downward_flux.assign_attrs(
        units="kg/s/m**2",
        long_name="Implied surface precipitation rate computed as E-<Q2>",
    )
    ds["Q2_flux"] = net_flux.assign_attrs(
        units="kg/s/m**2", long_name="Net flux of moisture"
    )
    ds["implied_surface_precipitation_rate"] = surface_downward_flux
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
    surface_upward_flux = ds[LHF] + ds[SHF] + ds[USW_SFC] + ds[ULW_SFC]
    Qm = _flux_to_tendency(
        ds["Qm_flux"],
        ds["implied_downward_radiative_flux_at_surface"],
        surface_upward_flux,
        ds[DELP],
    )
    ds["Qm"] = Qm.assign_attrs(units="W/kg")
    return ds


@register(
    ["Q2_flux", "implied_surface_precipitation_rate", DELP, LHF], ["Q2"],
)
def Q2_tendency_from_Q2_flux(ds):
    surface_upward_flux = vcm.latent_heat_flux_to_evaporation(ds[LHF])
    Q2 = _flux_to_tendency(
        ds["Q2_flux"],
        ds["implied_surface_precipitation_rate"],
        surface_upward_flux,
        ds[DELP],
    )
    ds["Q2"] = Q2.assign_attrs(units="kg/kg/s")
    return ds


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
    ["implied_downward_radiative_flux_at_surface"],
)
def implied_downward_radiative_flux_at_surface(
    ds, rectify=True, include_temperature_nudging=True
):
    """Assuming <Qm> = SHF + LHF + R_net + <T_nudge>."""
    toa_net_flux = ds[DSW_TOA] - ds[USW_TOA] - ds[ULW_TOA]
    if include_temperature_nudging:
        toa_net_flux += ds[COL_T_NUDGE]
    surface_upward_flux = ds[LHF] + ds[SHF] + ds[USW_SFC] + ds[ULW_SFC]
    surface_downward_flux = _tendency_to_implied_surface_downward_flux(
        ds["Qm"], toa_net_flux, surface_upward_flux, ds[DELP], dim="z", rectify=rectify
    )
    surface_downward_flux = surface_downward_flux.assign_attrs(
        units="W/m**2",
        long_name="Implied downward radiative flux from <Qm> budget closure",
    )
    ds["implied_downward_radiative_flux_at_surface"] = surface_downward_flux
    return ds


@register(["Q2", DELP, LHF], ["implied_surface_precipitation_rate"])
def implied_surface_precipitation_rate(ds, rectify=True):
    """Assuming <Q2> = E-P."""
    evaporation = vcm.latent_heat_flux_to_evaporation(ds[LHF])
    implied_precip = _tendency_to_implied_surface_downward_flux(
        ds["Q2"],
        xr.zeros_like(ds[LHF]),
        evaporation,
        ds[DELP],
        dim="z",
        rectify=rectify,
    )
    implied_precip = implied_precip.assign_attrs(
        units="kg/s/m**2",
        long_name="Implied surface precipitation rate computed as E-<Q2>",
    )
    ds["implied_surface_precipitation_rate"] = implied_precip
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
