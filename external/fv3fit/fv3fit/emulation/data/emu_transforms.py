"""
Array transforms to build onto our batching pipeline into training.
Generally assumes data to be in Sample x Feature shape.
"""

import abc
import logging
import os
import pickle
import numpy as np
import tensorflow as tf
import xarray as xr
from toolz.functoolz import curry
from typing import List, Mapping, Sequence, Tuple, Union

from .stacking import ArrayStacker

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
    return {
        varname: da.values
        for varname, da in dataset.items()
    }


@curry
def to_tensors(
    dataset: InputDataset, dtype: tf.DType = tf.float32
) -> TensorDataset:
    """Convert a dataset to tensors with specified dtype."""
    return {
        key: tf.convert_to_tensor(data, dtype=dtype)
        for key, data in dataset.items()
    }


@curry
def select_antarctic(dataset: xr.Dataset, sample_dim_name="sample") -> xr.Dataset:
    """Select only points below 60 S.  Requires 'latitude' in dataset."""

    mask = dataset["latitude"] < -np.deg2rad(60)
    dataset = dataset.isel({sample_dim_name: mask})

    return dataset


@curry
def standardize(std_info: StandardizeInfo, dataset: InputDataset):
    """
    Standardize data by removing the mean and scaling by a factor.
    """
    standardized = {}
    for varname in dataset:
        mean, std = std_info[varname]
        standardized[varname] = (dataset[varname] - mean) / std
    return standardized


@curry
def unstandardize(std_info: StandardizeInfo, dataset: InputDataset):
    """
    Unstandardize data by scaling by a factor and adding the mean.
    """
    unstandardized = {}
    for varname in dataset:
        mean, std = std_info[varname]
        unstandardized[varname] = (dataset[varname] * std) + mean
    return unstandardized


@curry
def stack_io(
    X_stacker: ArrayStacker,
    y_stacker: ArrayStacker,
    dataset: ArrayDataset
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Use stackers to combine dataset variables along the feature dimension.
    Groups dataset into stacked input and output data for training.
    """

    inputs = X_stacker.stack(dataset)
    outputs = y_stacker.stack(dataset)

    return inputs, outputs


@curry
def group_inputs_outputs(
    input_variables: List[str],
    output_variables: List[str],
    dataset: AnyDataset
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
    subselection_map: Mapping[str, slice],
    dataset: Union[ArrayDataset, TensorDataset]
) -> Union[ArrayDataset, TensorDataset]:
    """
    Subselect from the feature dimension if specified in the map.
    """

    new_ds = {}
    for vname, arr in dataset.items():
        subselect_slice = subselection_map.get(vname, None)
        if subselect_slice is not None:
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
