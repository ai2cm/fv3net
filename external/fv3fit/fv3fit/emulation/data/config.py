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
    Standard input pipeline that goes from xarray dataset to grouped
    X, y tuples of arrays/tensors per variable

    Args
        input_variables: Variables to include as inputs for training
        output_variables: Variables to include as targets for training
        antarctic_only: Limit data to < 60 S.  Requires latitude exists
            as a field in the dataset
        use_tensors: Converts data to float32 tensors instead of numpy arrays
        vertical_subselection: Limit the feature dimension of a variable
            to a specified range. Loaded in as slices from a 2 or 3 item
            sequence.
        from_netcdf_path: Prepend a netcdf opening transform (works on
            local/remote) to get xarray datasets from input path

    Example
        Yaml file example::
            
            from_netcdf_path: true
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
    from_netcdf_path: bool = True

    @classmethod
    def from_dict(cls, d: Dict):

        if "vertical_subselections" in d:
            d["vertical_subselections"] = _convert_map_sequences_to_slices(
                d["vertical_subselections"]
            )

        return dacite.from_dict(cls, d)

    def __call__(self, item: Any) -> Any:
        transform_pipeline = self._get_pipeline_from_config()
        return transform_pipeline(item)

    def _get_pipeline_from_config(self):

        transform_funcs = []

        if self.from_netcdf_path:
            transform_funcs.append(transforms.open_remote_nc)

        if self.antarctic_only:
            transform_funcs.append(transforms.select_antarctic)

        if self.use_tensors:
            transform_funcs.append(transforms.to_tensors)
        else:
            transform_funcs.append(transforms.to_ndarrays)

        transform_funcs.append(transforms.maybe_expand_feature_dim)

        if self.vertical_subselections is not None:
            transform_funcs.append(
                transforms.maybe_subselect_feature_dim(self.vertical_subselections)
            )

        transform_funcs.append(
            transforms.group_inputs_outputs(self.input_variables, self.output_variables)
        )

        return compose_left(*transform_funcs)
