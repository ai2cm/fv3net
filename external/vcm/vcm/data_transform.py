import dataclasses
from toolz.functoolz import curry
from typing import Any, Callable, Literal, Mapping, MutableMapping, Sequence, Set
import xarray as xr
import vcm


DATA_TRANSFORM_REGISTRY: MutableMapping[str, Mapping[str, Any]] = {}

TransformName = Literal[
    "q1_from_qm_q2", "qm_from_q1_q2", "q1_from_dQ1_pQ1", "q2_from_dQ2_pQ2",
]


@curry
def register(
    name: str,
    inputs: Sequence[str],
    outputs: Sequence[str],
    func: Callable[..., xr.Dataset],
):
    if name in DATA_TRANSFORM_REGISTRY:
        raise ValueError(f"Function {name} has already been added to registry.")
    DATA_TRANSFORM_REGISTRY[name] = {
        "func": func,
        "inputs": inputs,
        "outputs": outputs,
    }
    return func


@register("qm_from_q1_q2", ["Q1", "Q2"], ["Qm"])
def qm_from_q1_q2(ds):
    ds["Qm"] = vcm.moist_static_energy_tendency(ds["Q1"], ds["Q2"])
    return ds


@register("q1_from_qm_q2", ["Qm", "Q2"], ["Q1"])
def q1_from_qm_q2(ds):
    ds["Q1"] = vcm.temperature_tendency(ds["Qm"], ds["Q2"])
    return ds


@register("q1_from_dQ1_pQ1", ["dQ1", "pQ1"], ["Q1"])
def q1_from_dQ1_pQ1(ds):
    ds["Q1"] = ds["dQ1"] + ds["pQ1"]
    return ds


@register("q2_from_dQ2_pQ2", ["dQ2", "pQ2"], ["Q2"])
def q2_from_dQ2_pQ2(ds):
    ds["Q2"] = ds["dQ2"] + ds["pQ2"]
    return ds


@dataclasses.dataclass
class DataTransform:
    name: TransformName
    kwargs: dict = dataclasses.field(default_factory=dict)

    def apply(self, ds: xr.Dataset) -> xr.Dataset:
        func = DATA_TRANSFORM_REGISTRY[self.name]["func"]
        ds = func(ds, **self.kwargs)
        return ds

    @property
    def input_variables(self) -> Sequence[str]:
        return DATA_TRANSFORM_REGISTRY[self.name]["inputs"]

    @property
    def output_variables(self) -> Sequence[str]:
        return DATA_TRANSFORM_REGISTRY[self.name]["outputs"]


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
