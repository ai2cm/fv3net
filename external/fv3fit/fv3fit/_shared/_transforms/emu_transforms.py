"""
Array transforms to build onto our batching pipeline into training.
Generally assumes data to be in Sample x Feature shape.
"""

import logging
import numpy as np
import xarray as xr
from toolz.functoolz import curry
from typing import Mapping, Tuple

from .stacking import ArrayStacker

logger = logging.getLogger(__name__)

Mean = np.ndarray
StandardFactor = np.ndarray
StandardizeInfo = Mapping[str, Tuple[Mean, StandardFactor]]
FeatureSubselection = Mapping[str, slice]
ArrayDataset = Mapping[str, np.ndarray]


def extract_ds_arrays(dataset: xr.Dataset) -> ArrayDataset:
    # todo this requires dataset op, maybe switch all to dict of ndarrays
    sample_size = dataset.sizes["sample"]
    return {
        varname: da.values.reshape(sample_size, -1)
        for varname, da in dataset.items()
    }
  

@curry
def standardize(std_info: StandardizeInfo, dataset: ArrayDataset):
    """
    Standardize data by removing the mean and scaling by a factor.

    Args:
        std_info: Mean and standardization factor for each variable. 
        dataset: Variables to be standardized
    """
    standardized = {}
    for varname in dataset:
        mean, std = std_info[varname]
        standardized[varname] = (dataset[varname] - mean) / std
    return standardized


@curry
def unstandardize(std_info: StandardizeInfo, dataset: ArrayDataset):
    """
    Unstandardize data by scaling by a factor and adding the mean.

    Args:
        std_info: Mean and standardization factor for each variable. 
        dataset: Variables to be standardized
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
    
    inputs = X_stacker.stack(dataset)
    outputs = y_stacker.stack(dataset)
    
    return inputs, outputs
