"""
Trying to update old transforms that don't easly fit
into the new from config style yet.
"""

import abc
import os
import pickle
import numpy as np
import tensorflow as tf
import xarray as xr
from toolz.functoolz import curry
from typing import Mapping, Tuple, Union

from .stacking import ArrayStacker


InputDataset = Mapping[str, Union[np.ndarray, xr.DataArray]]
NumericContainer = Union[np.ndarray, xr.DataArray, tf.Tensor]
ArrayDataset = Mapping[str, np.ndarray]
StandardizeInfo = Mapping[str, Tuple[NumericContainer, NumericContainer]]
AnyDataset = Mapping[str, NumericContainer]


class DumpableTransform(abc.ABC):
    """
    Base class for transforms requiring some extra
    state information to operate on a dataset with operations
    to serialize to disk.
    """

    def __init__(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def __call__(self, dataset: AnyDataset) -> AnyDataset:
        pass

    @abc.abstractmethod
    def dump(self, path: str):
        pass

    @abc.abstractclassmethod
    def load(self, path: str):
        pass


class Standardize(DumpableTransform):

    OUTPUT_FILE = "standardization_info.pkl"

    def __init__(self, std_info: StandardizeInfo):
        self.std_info = std_info

    def __call__(self, dataset: AnyDataset) -> AnyDataset:
        standardized = {}
        for varname in dataset:
            mean, std = self.std_info[varname]
            standardized[varname] = (dataset[varname] - mean) / std
        return standardized

    def dump(self, path: str):
        with open(os.path.join(path, self.OUTPUT_FILE), "wb") as f:
            pickle.dump(self.std_info, f)


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
    X_stacker: ArrayStacker, y_stacker: ArrayStacker, dataset: ArrayDataset
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Use stackers to combine dataset variables along the feature dimension.
    Groups dataset into stacked input and output data for training.
    """

    inputs = X_stacker.stack(dataset)
    outputs = y_stacker.stack(dataset)

    return inputs, outputs
