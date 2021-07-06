import dacite
import dataclasses
import xarray
import numpy as np
import tensorflow as tf
from toolz.functoolz import compose_left
from typing import Any, Dict, Mapping, Sequence, Tuple, Union

from . import transforms

Dataset = Union[xarray.Dataset, Mapping[str, np.ndarray], Mapping[str, tf.Tensor]]
XyTensors = Tuple[Tuple[tf.Tensor], Tuple[tf.Tensor]]


@dataclasses.dataclass
class _TransformConfigItem:
    # put a note about the args order relevance
    name: str
    args: Union[Sequence[Any], Mapping[str, Any]] = dataclasses.field(default_factory=list)
    kwargs: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.args, Mapping):
            self.args = list(self.args.values())

    @classmethod
    def from_dict(cls, item_info):
        return dacite.from_dict(data_class=cls, data=item_info)

    def load_transform(self):
        func = getattr(transforms, self.name)
        return func(*self.args, **self.kwargs)


def _load_transforms(transforms_to_load: Sequence[_TransformConfigItem]):

    loaded_transforms = [
        transform_info.load_transform()
        for transform_info in transforms_to_load
    ]
    return compose_left(*loaded_transforms)


TransformItemSpec = Union[Mapping, _TransformConfigItem]
TransformConfigSpec = Dict[str, Sequence[TransformItemSpec]]


@dataclasses.dataclass
class TransformConfig:
    """
    Specify exact transform pipeline

    example
    """
    transforms: Sequence[_TransformConfigItem] = dataclasses.field(default_factory=list)

    @staticmethod
    def _initialize_custom_transforms(d: TransformConfigSpec):
        v = d.pop("transforms")
        initialized = [_TransformConfigItem.from_dict(item_info) for item_info in v]
        d["transforms"] = initialized
        return d

    @classmethod
    def from_dict(cls, d):
        d = cls._initialize_custom_transforms(d)
        return dacite.from_dict(data_class=cls, data=d)

    def get_transform_func(self):
        if self.transforms:
            return _load_transforms(self.transforms)
        else:
            return lambda x: x


@dataclasses.dataclass
class InputTransformConfig(TransformConfig):
    """
    Standard input pipeline that goes from xarray dataset to grouped
    X, y tuples of arrays/tensors per variable
    """
    input_variables: Sequence[str] = dataclasses.field(default_factory=list)
    output_variables: Sequence[str] = dataclasses.field(default_factory=list)
    antarctic_only: bool = False
    use_tensors: bool = True
    vertical_subselections: Union[Mapping[str, slice], None] = None
    transforms: Sequence[_TransformConfigItem] = dataclasses.field(default_factory=list)

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
            transform_funcs += _load_transforms(self.transforms)

        transform_funcs.append(
            transforms.group_inputs_outputs(
                self.input_variables,
                self.output_variables
            )
        )

        return compose_left(*transform_funcs)
