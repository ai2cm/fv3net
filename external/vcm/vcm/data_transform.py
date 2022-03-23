import dataclasses
import numpy as np
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
    "subsample_weights_from_latitude",
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


@register(["latitude"], ["subsample_weights"])
def subsample_weights_from_latitude(ds, exponent: float = 1.3):
    # weight by some power of inverse of pdf(latitude) to upsample columns
    # that are closer to poles
    # the default power of 1.5 increases weighting towards poles
    weights = 1.0 / np.cos(ds["latitude"]) ** exponent
    ds["subsample_weights"] = weights / weights.sum()
    return ds


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
            print(f"applying transform {transform.name}")
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
