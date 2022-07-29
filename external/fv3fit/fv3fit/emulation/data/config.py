import dataclasses
import logging
from typing import Any, Callable, Dict, List, Mapping, Optional, Set, Union

import tensorflow as tf
import dacite
import xarray
from fv3fit._shared import SliceConfig
from toolz.functoolz import pipe
import numpy as np

from fv3fit.emulation.transforms import (
    Difference,
    CloudWaterDiffPrecpd,
    MicrophysicsClasssesV1,
    MicrophysicsClassesV1OneHot,
    GscondRoute,
)
from . import transforms

logger = logging.getLogger(__name__)


# SimpleTransforms must be transforms rather than factories to ensure that
# transforms that are fit from data are not used here
SimpleTransforms = Union[
    Difference,
    CloudWaterDiffPrecpd,
    MicrophysicsClasssesV1,
    MicrophysicsClassesV1OneHot,
    GscondRoute,
]


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
        tensor_transforms: Transforms that will not be included in the
            serialized model artifact. Must be transforms (e.g. have forward and
            backward method) rather than factories to ensure that transforms trained
            with data are not used here.

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
    tensor_transforms: List[SimpleTransforms] = dataclasses.field(default_factory=list)

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
        variables = self.get_dataset_names(variables)
        return Pipeline(
            self._get_xarray_transforms(variables),
            self._get_array_like_transforms(),
            use_tensors=self.use_tensors,
        )

    def get_dataset_names(self, variables: Set[str]) -> Set[str]:
        # compute the required variable names
        for transform in self.tensor_transforms[::-1]:
            variables = transform.backward_names(variables)
        return variables

    def _get_array_like_transforms(self) -> List:
        # array-like dataset transforms
        transform_funcs = [transforms.expand_single_dim_data]
        if self.vertical_subselections is not None:
            transform_funcs.append(
                transforms.maybe_subselect_feature_dim(self.vert_sel_as_slices)
            )

        for transform in self.tensor_transforms:
            transform_funcs.append(transform.forward)  # type: ignore
        return transform_funcs

    def _get_xarray_transforms(
        self, variables: Set[str]
    ) -> List[Callable[[xarray.Dataset], xarray.Dataset]]:
        transform_funcs = []
        if self.antarctic_only:
            transform_funcs.append(transforms.select_antarctic)

        transform_funcs.append(
            transforms.derived_dataset(
                list(variables), tendency_timestep_sec=self.derived_microphys_timestep,
            )
        )

        return transform_funcs
