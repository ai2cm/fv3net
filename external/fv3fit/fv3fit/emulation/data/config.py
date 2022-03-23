import dataclasses
import logging
from typing import Any, Callable, Dict, List, Mapping, Optional, Set, Union

import tensorflow as tf
import dacite
import xarray
from fv3fit._shared import SliceConfig
from toolz.functoolz import pipe
import numpy as np

from . import transforms

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Pipeline:
    xarray_transforms: List[Callable[[xarray.Dataset], xarray.Dataset]]
    array_like_transforms: List[Any]
    use_tensors: bool = True

    def __call__(
        self, dataset: xarray.Dataset
    ) -> Mapping[str, Union[np.ndarray, tf.Tensor]]:
        transformed_dataset = pipe(dataset, *self.xarray_transforms)

        if self.use_tensors:
            array_like = transforms.to_tensors(transformed_dataset)
        else:
            array_like = transforms.to_ndarrays(transformed_dataset)

        return pipe(array_like, *self.array_like_transforms)


@dataclasses.dataclass
class TransformConfig:
    """
    Standard input pipeline that goes from an xarray dataset with data
    dimensions of [sample, feature] or [sample] to grouped
    X, y tuples of arrays/tensors per variable

    Args:
        antarctic_only: Limit data to < 60 S.  Requires latitude exists
            as a field in the dataset
        use_tensors: Converts data to float32 tensors instead of numpy arrays
        vertical_subselection: Limit the feature dimension of a variable
            to a specified range. Loaded in as slices from a 2 or 3 item
            sequence.

    Example:
        Yaml file example::

            antarctic_only: true
            use_tensors: true
            vertical_subselections:
              a:
                stop: 5
              b:
                start: 5
              c:
                start: 5
                stop: 15
                step: 2
    """

    antarctic_only: bool = False
    use_tensors: bool = True
    vertical_subselections: Optional[Mapping[str, SliceConfig]] = None
    derived_microphys_timestep: int = 900

    @classmethod
    def from_dict(cls, d: Dict):
        return dacite.from_dict(cls, d, config=dacite.Config(strict=True))

    def __post_init__(self):
        if self.vertical_subselections is not None:
            self.vert_sel_as_slices = {
                k: v.slice for k, v in self.vertical_subselections.items()
            }
        else:
            self.vert_sel_as_slices = None

    def get_pipeline(self, variables: Set[str]) -> Pipeline:
        """
        Args:
            variables: the variables required for training. Both inputs and outputs.

        Returns:
            conversion from dataset to dict of numpy or tensorflow tensors
        """
        return Pipeline(
            self._get_xarray_transforms(variables),
            self._get_array_like_transforms(),
            use_tensors=self.use_tensors,
        )

    def _get_array_like_transforms(self):
        # array-like dataset transforms
        transform_funcs = [transforms.expand_single_dim_data]
        if self.vertical_subselections is not None:
            transform_funcs.append(
                transform_funcs.maybe_subselect_feature_dim(self.vert_sel_as_slices)
            )
        return transform_funcs

    def _get_xarray_transforms(
        self, variables: Set[str]
    ) -> List[Callable[[xarray.Dataset], xarray.Dataset]]:
        transform_funcs = []

        if self.antarctic_only:
            transform_funcs.append(transforms.select_antarctic)

        transform_funcs.append(transforms.select_variables(variables))

        return transform_funcs
