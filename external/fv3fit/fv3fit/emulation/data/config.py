import dacite
import dataclasses
import logging
from toolz.functoolz import compose_left
from typing import Any, Dict, Mapping, Optional, Sequence

from . import transforms


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class SliceConfig:
    start: Optional[int] = None
    stop: Optional[int] = None
    step: Optional[int] = None

    @property
    def slice(self):
        return slice(self.start, self.stop, self.step)


@dataclasses.dataclass
class TransformConfig:
    """
    Standard input pipeline that goes from an xarray dataset with data
    dimensions of [sample, feature] or [sample] to grouped
    X, y tuples of arrays/tensors per variable

    Args:
        input_variables: Variables to include as inputs for training
        output_variables: Variables to include as targets for training
        antarctic_only: Limit data to < 60 S.  Requires latitude exists
            as a field in the dataset
        use_tensors: Converts data to float32 tensors instead of numpy arrays
        vertical_subselection: Limit the feature dimension of a variable
            to a specified range. Loaded in as slices from a 2 or 3 item
            sequence.

    Example:
        Yaml file example::

            input_variables: ["a", "b"]
            output_variables: ["c", "d"]
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

    input_variables: Sequence[str] = dataclasses.field(default_factory=list)
    output_variables: Sequence[str] = dataclasses.field(default_factory=list)
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

    def __call__(self, item: Any) -> Any:
        transform_pipeline = self._get_pipeline_from_config()
        return transform_pipeline(item)

    def _get_pipeline_from_config(self):

        transform_funcs = []

        # xarray transforms

        if self.antarctic_only:
            transform_funcs.append(transforms.select_antarctic)

        transform_funcs.append(
            transforms.derived_dataset(
                list(self.input_variables) + list(self.output_variables),
                tendency_timestep_sec=self.derived_microphys_timestep,
            )
        )

        if self.use_tensors:
            transform_funcs.append(transforms.to_tensors)
        else:
            transform_funcs.append(transforms.to_ndarrays)

        # array-like dataset transforms
        transform_funcs.append(transforms.expand_single_dim_data)

        if self.vertical_subselections is not None:
            transform_funcs.append(
                transforms.maybe_subselect_feature_dim(self.vert_sel_as_slices)
            )

        # final transform to grouped X, y tuples
        transform_funcs.append(
            transforms.group_inputs_outputs(self.input_variables, self.output_variables)
        )

        return compose_left(*transform_funcs)
