import dataclasses
import xarray
import numpy as np
import tensorflow as tf
from toolz.functoolz import compose_left
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union

from . import emu_transforms as transforms

TransformKwargs = Mapping[str, Any]
TransformItem = Dict[str, TransformKwargs]

Dataset = Union[xarray.Dataset, Mapping[str, np.ndarray], Mapping[str, tf.Tensor]]
XyTensors = Tuple[Tuple[tf.Tensor], Tuple[tf.Tensor]]


def load_transforms(transforms_to_load: Sequence[TransformItem]):
    
    loaded_transforms = []
    for transform_info in transforms_to_load:
        target_transform, transform_kwargs = transform_info.popitem()
        transform_func = getattr(transforms, target_transform)
        # TODO: I use kwargs in yaml for readability, but don't know how
        #       to do partials by assiging the keyword arguments as is
        #       For now, the ordering of values matters for partial
        transform_func = transform_func(*transform_kwargs.values())
        loaded_transforms.append(transform_func)

    return loaded_transforms


@dataclasses.dataclass
class TransformConfig:
    """
    Specify exact transform pipeline

    example
    """
    transforms: List[TransformItem] = dataclasses.field(default_factory=list)

    def get_transform_func(self):
        if self.transforms:
            return compose_left(load_transforms(self.transforms))
        else:
            return lambda x: x


@dataclasses.dataclass
class InputTransformConfig:
    """
    Standard input pipeline that goes from xarray dataset to grouped
    X, y tuples of arrays/tensors per variable
    """
    transforms: List[TransformItem] = dataclasses.field(default_factory=list)
    input_variables: List[str] = dataclasses.field(default_factory=list)
    output_variables: List[str] = dataclasses.field(default_factory=list)
    antarctic_only: bool = False
    use_tensors: bool = True
    vertical_subselections: Union[Mapping[str, slice], None] = None

    def get_transform_func(self):

        transform_funcs = []

        if self.antarctic_only:
            transform_funcs.append(transforms.select_antarctic)

        if self.use_tensors:
            transform_funcs.append(transforms.to_tensors)
        else:
            transform_funcs.append(transforms.to_ndarrays)

        transform_funcs.append(transforms.maybe_expand_feature_dim)

        if self.vertical_subselections is not None:
            transform_funcs.append(transforms.maybe_subselect(self.vertical_subselections))

        if self.transforms:
            transform_funcs += load_transforms(self.transforms)

        transform_funcs.append(
            transforms.group_inputs_outputs(
                self.input_variables,
                self.output_variables
            )
        )

        return compose_left(*transform_funcs)
