"""
Array transforms to build onto our batching pipeline into training.
Generally assumes data to be in Sample x Feature shape.
"""

import logging
import numpy as np
import tensorflow as tf
from vcm.derived_mapping import DerivedMapping
import xarray as xr
from toolz.functoolz import curry
from typing import Hashable, Mapping, Sequence, Union

from vcm import get_fs, open_remote_nc


logger = logging.getLogger(__name__)

NumericContainer = Union[np.ndarray, xr.DataArray, tf.Tensor]
ArrayDataset = Mapping[Hashable, np.ndarray]
TensorDataset = Mapping[Hashable, tf.Tensor]
AnyDataset = Mapping[Hashable, NumericContainer]


def open_netcdf_dataset(path: str) -> xr.Dataset:
    """Open a netcdf from a local/remote path"""

    fs = get_fs(path)
    data = open_remote_nc(fs, path)

    return data


@curry
def derived_dataset(
    all_variables: Sequence[str], dataset: AnyDataset, tendency_timestep_sec: int = 900
):

    derived = DerivedMapping(dataset, microphys_timestep_sec=tendency_timestep_sec)
    dataset = derived.dataset(all_variables)

    return dataset


def to_ndarrays(dataset: AnyDataset) -> ArrayDataset:
    """Convert a dataset to ndarrays with a specified dtype."""
    logger.debug("Converting dataset to ndarray dataset")
    return {varname: np.array(da) for varname, da in dataset.items()}


@curry
def to_tensors(dataset: AnyDataset, dtype: tf.DType = tf.float32) -> TensorDataset:
    """Convert a dataset to tensors with specified dtype."""
    logger.debug("Converting dataset to tensor dataset")
    return {
        key: tf.convert_to_tensor(data, dtype=dtype) for key, data in dataset.items()
    }


@curry
def select_antarctic(dataset: xr.Dataset, sample_dim_name="sample") -> xr.Dataset:
    """
    Select only points below 60 S.  Requires 'latitude' in dataset and expects
    units in radians
    """

    logger.debug("Reducing samples to antarctic points (<60S) only")
    mask = dataset["latitude"] < -np.deg2rad(60)
    dataset = dataset.isel({sample_dim_name: mask})

    return dataset


@curry
def maybe_subselect_feature_dim(
    subselection_map: Mapping[Hashable, slice],
    dataset: Union[ArrayDataset, TensorDataset],
) -> Union[ArrayDataset, TensorDataset]:
    """
    Subselect from the feature dimension if specified in the map.
    """
    new_ds = {}
    for vname, arr in dataset.items():
        if vname in subselection_map:
            subselect_slice = subselection_map[vname]
            logger.debug(
                f"Subselecting features from {vname} using slice {subselect_slice}"
            )
            arr = arr[..., subselect_slice]

        new_ds[vname] = arr

    return new_ds


def expand_single_dim_data(
    dataset: Union[ArrayDataset, TensorDataset]
) -> Union[ArrayDataset, TensorDataset]:
    """Expand a feature dimension for single-dimension data"""

    new_ds = {}
    for key, data in dataset.items():
        if len(data.shape) == 1:
            logger.debug(f"Expanding feature dimension for {key}")
            data = data[:, None]

        new_ds[key] = data

    return new_ds
