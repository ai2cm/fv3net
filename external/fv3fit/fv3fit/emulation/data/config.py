import dacite
import dataclasses
import logging
import xarray
import yaml
import numpy as np
import tensorflow as tf
from toolz.functoolz import compose_left
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple, Union

from . import transforms


logger = logging.getLogger(__name__)


class SliceLoader(yaml.SafeLoader):
    """
    Extended safe yaml loader to interpret slices from a sequence.

    Example
    -------

    The vertical subselection transform yaml::

        name: maybe_subselect
        args:
          - specific_humidity: !!python/slice [15, null]
            tendency_of_sphum: !!python/slice [15, null]

    Loading the yaml file::

        with open("yaml_file_with_slices.yaml", "r") as f:
            yaml.load(f, Loader=SliceLoader)

    """

    def construct_python_slice(self, node):
        return slice(*self.construct_sequence(node))


SliceLoader.add_constructor(
    "tag:yaml.org,2002:python/slice", SliceLoader.construct_python_slice
)


Dataset = Union[xarray.Dataset, Mapping[str, np.ndarray], Mapping[str, tf.Tensor]]
XyTensors = Tuple[Tuple[tf.Tensor], Tuple[tf.Tensor]]


@dataclasses.dataclass
class _TransformConfigItem:
    """
    Specification of a transform function and any
    args and/or kwargs necessary to curry it into a
    function taking a single argument.

    Args
    ----
        name: Curry-able function name in the transforms module
        args: Arguments to provide into curried function such that
            the resulting call signature is func(dataset).  Note:
            This can be specified as a mapping but it will be converted
            to a list in the order supplied.
        kwargs: Keyword arguments to provide to the curried function.

    Example
    -------
    Example yaml for grouped inputs/outputs transform::

        name: group_inputs_outputs
        args:
          - ["field1", "field2"]
          - ["field3"]

    Equivalent yaml for grouped inputs/outputs transform::

        name: group_inputs_outputs
        args:
          input_variables: ["field1", "field2"]
          output_variables: ["field3"]
    """

    name: str
    args: Union[Sequence[Any], Mapping[str, Any]] = dataclasses.field(
        default_factory=list
    )
    kwargs: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.args, Mapping):
            self.args = list(self.args.values())

    @classmethod
    def from_dict(cls, item_info):
        return dacite.from_dict(data_class=cls, data=item_info)

    def load_transform_func(self) -> Callable:
        func = getattr(transforms, self.name)
        partial_func = func(*self.args, **self.kwargs)

        if not callable(partial_func):
            raise TypeError(
                f"Loaded transform for {self.name} from config is no"
                " longer callable. Check that partial arguments match"
                " the signature."
            )
        return partial_func


def _load_transforms(transforms_to_load: Sequence[_TransformConfigItem]):

    loaded_transforms = [
        transform_info.load_transform_func() for transform_info in transforms_to_load
    ]
    logger.debug(
        f"Loaded transform sequence: {[xfm.name for xfm in transforms_to_load]}"
    )
    return compose_left(*loaded_transforms)


TransformItemSpec = Union[Mapping, _TransformConfigItem]
TransformConfigSpec = Dict[str, Sequence[TransformItemSpec]]


@dataclasses.dataclass
class TransformConfig:
    """
    Specify a custom transform pipeline for data

    Args
    ----
        transforms: Sequence of transform configurations to combine in order

    Example
    -------
    Example yaml converting dataset to tensors and grouping::

        transforms:
          - name: to_tensors
          - name: group_inputs_outputs
            args:
              - ["field1", "field2"]
              - ["field3"]
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
        if "transforms" in d:
            d = cls._initialize_custom_transforms(d)
        return dacite.from_dict(data_class=cls, data=d)

    def get_transform_pipeline(self):
        if self.transforms:
            return _load_transforms(self.transforms)
        else:
            return lambda x: x


@dataclasses.dataclass
class InputTransformConfig(TransformConfig):
    """
    Standard input pipeline that goes from xarray dataset to grouped
    X, y tuples of arrays/tensors per variable

    Args
    ----
        input_variables: Variables to include as inputs for training
        output_variables: Variables to include as targets for training
        antarctic_only: Limit data to < 60 S.  Requires latitude exists
            as a field in the dataset
        use_tensors: Converts data to float32 tensors instead of numpy arrays
        vertical_subselection: Limit the feature dimension of a variable
            to a specified range. Loaded in as slices from a 2 or 3 item
            sequence.
        transforms: Sequence of extra transform configurations to combine
            in order. Inserted just before input/output grouping function.

    Example
    -------
    Yaml file example::

        input_variables: ["a", "b"]
        output_variables: ["c", "d"]
        antarctic_only: true
        use_tensors: true
        vertical_subselections:
          a: !!python/slice [5]
          b: !!python/slice [5, None]
          c: !!python/slice [5, 20, 2]

    Note that the slice loading in yaml requires using the SliceLoader defined
    in this module.
    """

    input_variables: Sequence[str] = dataclasses.field(default_factory=list)
    output_variables: Sequence[str] = dataclasses.field(default_factory=list)
    antarctic_only: bool = False
    use_tensors: bool = True
    vertical_subselections: Union[Mapping[str, slice], None] = None

    def get_transform_pipeline(self):

        transform_funcs = []

        if self.antarctic_only:
            transform_funcs.append(transforms.select_antarctic)

        if self.use_tensors:
            transform_funcs.append(transforms.to_tensors)
        else:
            transform_funcs.append(transforms.to_ndarrays)

        transform_funcs.append(transforms.maybe_expand_feature_dim)

        if self.vertical_subselections is not None:
            transform_funcs.append(
                transforms.maybe_subselect(self.vertical_subselections)
            )

        if self.transforms:
            transform_funcs += _load_transforms(self.transforms)

        transform_funcs.append(
            transforms.group_inputs_outputs(self.input_variables, self.output_variables)
        )

        return compose_left(*transform_funcs)
