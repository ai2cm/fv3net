import dataclasses
import xarray
import numpy as np
import tensorflow as tf
from toolz.functoolz import compose_left
from typing import Any, Callable, Dict, List, Mapping, Tuple, Union


TransformKwargs = Mapping[str, Any]  
TransformConfig = Dict[str, TransformKwargs]  

Dataset = Union[xarray.Dataset, Mapping[str, np.ndarray], Mapping[str, tf.Tensor]]
XyTensors = Tuple[Tuple[tf.Tensor], Tuple[tf.Tensor]]


@dataclasses.dataclass
class CustomTransformConfig:
    transforms: List[TransformConfig] = dataclasses.field(default_factory=list)

    def get_transform_func(self):
        return load_input_transforms(self)


@dataclasses.dataclass
class InputTransformConfig:
    input_variables: List[str]
    output_variables: List[str]
    transforms: List[TransformConfig] = dataclasses.field(default_factory=list)

    def get_transform_func(self):
        return load_input_transforms(self)

def load_input_transforms(config: DataInputConfig) -> Callable[[Dataset], XyTensors]:
    transforms = []
    for transform_info in config.transforms:
        target_transform, transform_kwargs = transform_info.popitem()
        if target_transform == "group_inputs_outputs":
            raise ValueError(
                "group_inputs_outputs is always included at the end."
                " Please remove entry from the 'transforms' section of the config."
            )
        transform_func = getattr(transform_mod, target_transform)
        # TODO: I use kwargs in yaml for readability, but don't know how
        #       to do partials by assiging the keyword arguments as is
        #       For now, the ordering of values matters for partial
        transform_func = transform_func(*transform_kwargs.values())
        transforms.append(transform_func)

    group_io_tuples = transform_mod.group_inputs_outputs(
        config.input_variables, config.output_variables
    )
    transforms.append(group_io_tuples)

    return compose_left(*transforms)