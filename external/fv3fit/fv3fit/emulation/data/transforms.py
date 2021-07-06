"""
Array transforms to build onto our batching pipeline into training.
Generally assumes data to be in Sample x Feature shape.
"""

import logging
import numpy as np
import tensorflow as tf
import xarray as xr
from toolz.functoolz import curry
from typing import Mapping, Sequence, Tuple, Union


logger = logging.getLogger(__name__)

NumericContainer = Union[np.ndarray, xr.DataArray, tf.Tensor]
StandardizeInfo = Mapping[str, Tuple[NumericContainer, NumericContainer]]
FeatureSubselection = Mapping[str, slice]
ArrayDataset = Mapping[str, np.ndarray]
TensorDataset = Mapping[str, tf.Tensor]
InputDataset = Mapping[str, Union[np.ndarray, xr.DataArray]]
AnyDataset = Mapping[str, NumericContainer]


def to_ndarrays(dataset: xr.Dataset) -> ArrayDataset:
    """Convert a dataset to ndarrays with a specified dtype."""
    return {varname: da.values for varname, da in dataset.items()}


@curry
def to_tensors(dataset: InputDataset, dtype: tf.DType = tf.float32) -> TensorDataset:
    """Convert a dataset to tensors with specified dtype."""
    return {
        key: tf.convert_to_tensor(data, dtype=dtype) for key, data in dataset.items()
    }


@curry
def select_antarctic(dataset: xr.Dataset, sample_dim_name="sample") -> xr.Dataset:
    """
    Select only points below 60 S.  Requires 'latitude' in dataset and expects
    units in radians
    """

    mask = dataset["latitude"] < -np.deg2rad(60)
    dataset = dataset.isel({sample_dim_name: mask})

    return dataset


@curry
def group_inputs_outputs(
    input_variables: Sequence[str], output_variables: Sequence[str], dataset: AnyDataset
) -> Tuple[Sequence[NumericContainer], Sequence[NumericContainer]]:
    """
    Group input and output variables into separate tuples where each item
    is a value associated with a variable in the order of the mapping
    keys.
    """

    inputs_ = tuple([dataset[key] for key in input_variables])
    outputs_ = tuple([dataset[key] for key in output_variables])

    return inputs_, outputs_


@curry
def maybe_subselect(
    subselection_map: Mapping[str, slice], dataset: Union[ArrayDataset, TensorDataset]
) -> Union[ArrayDataset, TensorDataset]:
    """
    Subselect from the feature dimension if specified in the map.
    """

    new_ds = {}
    for vname, arr in dataset.items():
        if vname in subselection_map:
            subselect_slice = subselection_map[vname]
            arr = arr[..., subselect_slice]

        new_ds[vname] = arr

    return new_ds


@curry
def maybe_expand_feature_dim(
    dataset: Union[ArrayDataset, TensorDataset]
) -> Union[ArrayDataset, TensorDataset]:
    """Expand a feature dimension for single-dimension data"""

    new_ds = {}
    for key, data in dataset.items():
        if len(data.shape) == 1:
            data = data[:, None]

        new_ds[key] = data

    return new_ds
