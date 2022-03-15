import dataclasses
from toolz.functoolz import curry
from typing import Callable, Literal, Sequence, MutableMapping
import xarray as xr
import vcm

DATA_TRANSFORM_FUNCTIONS: MutableMapping[str, Callable[..., xr.Dataset]] = {}


@curry
def register(
    name: str, func: Callable[..., xr.Dataset], registry=DATA_TRANSFORM_FUNCTIONS
):
    if name in registry:
        raise ValueError(f"Function {name} has already been added to registry.")
    registry[name] = func
    return func


@register("qm_from_q1_q2")
def qm_from_q1_q2(ds, temperature_dependent_latent_heat=False):
    if temperature_dependent_latent_heat:
        Qm = vcm.moist_static_energy_tendency(
            ds["Q1"], ds["Q2"], temperature=ds["air_temperature"]
        )
    else:
        Qm = vcm.moist_static_energy_tendency(ds["Q1"], ds["Q2"])
    ds["Qm"] = Qm
    return ds


@register("q1_from_qm_q2")
def q1_from_qm_q2(ds, temperature_dependent_latent_heat=False):
    if temperature_dependent_latent_heat:
        Q1 = vcm.temperature_tendency(
            ds["Qm"], ds["Q2"], temperature=ds["air_temperature"]
        )
    else:
        Q1 = vcm.temperature_tendency(ds["Qm"], ds["Q2"])
    ds["Q1"] = Q1
    return ds


@register("q1_from_dQ1_pQ1")
def q1_from_dQ1_pQ1(ds):
    ds["Q1"] = ds["dQ1"] + ds["pQ1"]
    return ds


@register("q2_from_dQ2_pQ2")
def q2_from_dQ2_pQ2(ds):
    ds["Q2"] = ds["dQ2"] + ds["pQ2"]
    return ds


@dataclasses.dataclass
class DataTransformConfig:
    name: Literal[
        "q1_from_qm_q2", "qm_from_q1_q2", "q1_from_dQ1_pQ1", "q2_from_dQ2_pQ2"
    ]
    kwargs: dict = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class DataTransform:
    config: DataTransformConfig

    def apply(self, ds: xr.Dataset) -> xr.Dataset:
        func = DATA_TRANSFORM_FUNCTIONS[self.config.name]
        ds = func(ds, **self.config.kwargs)
        return ds


@dataclasses.dataclass
class ChainedDataTransform:
    config: Sequence[DataTransformConfig]

    def apply(self, ds: xr.Dataset) -> xr.Dataset:
        for transform_config in self.config:
            ds = DataTransform(transform_config).apply(ds)
        return ds
