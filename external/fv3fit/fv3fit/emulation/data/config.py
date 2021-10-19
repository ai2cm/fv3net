import dacite
import dataclasses
import logging
from toolz.functoolz import compose_left
from typing import Any, Dict, Mapping, Optional, Sequence, Union

from . import transforms


logger = logging.getLogger(__name__)


def _sequence_to_slice(seq: Sequence[Union[None, int]]):

    if not seq:
        slice_ = slice(None)
    elif len(seq) > 3:
        raise ValueError(
            "Converting sequence to slice failed. Expected maximum of 3 values"
            f" but received {seq}."
        )
    else:
        slice_ = slice(*seq)

    return slice_


def _convert_map_sequences_to_slices(map_: Mapping[str, Sequence[int]]):

    return {key: _sequence_to_slice(seq) for key, seq in map_.items()}


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
              a: [5]
              b: [5, None]
              c: [5, 20, 2]
    """

    input_variables: Sequence[str] = dataclasses.field(default_factory=list)
    output_variables: Sequence[str] = dataclasses.field(default_factory=list)
    antarctic_only: bool = False
    use_tensors: bool = True
    vertical_subselections: Optional[Mapping[str, slice]] = None
    derived_microphys_timestep: int = 900

    @classmethod
    def from_dict(cls, d: Dict):

        subselect_key = "vertical_subselections"
        if subselect_key in d:
            d[subselect_key] = _convert_map_sequences_to_slices(d[subselect_key])

        return dacite.from_dict(cls, d)

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
                transforms.maybe_subselect_feature_dim(self.vertical_subselections)
            )

        # final transform to grouped X, y tuples
        transform_funcs.append(
            transforms.group_inputs_outputs(self.input_variables, self.output_variables)
        )

        return compose_left(*transform_funcs)
