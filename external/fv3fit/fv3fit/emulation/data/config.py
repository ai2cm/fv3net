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


Dataset = Union[xarray.Dataset, Mapping[str, np.ndarray], Mapping[str, tf.Tensor]]
XyTensors = Tuple[Tuple[tf.Tensor], Tuple[tf.Tensor]]


@dataclasses.dataclass
class InputTransformConfig:
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

    Loading the combined transform function::

        with open("config.yaml", "r") as f:
            config = yaml.load(f, Loader=SliceLoader)

        input_transforms = InputTransformConfig.from_dict(config)
        pipeline = input_transforms.get_transform_pipeline()
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

        transform_funcs.append(
            transforms.group_inputs_outputs(self.input_variables, self.output_variables)
        )

        return compose_left(*transform_funcs)
