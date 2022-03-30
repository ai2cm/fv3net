import collections
from toolz.functoolz import curry
from typing import (
    Hashable,
    List,
    Dict,
    Tuple,
    Sequence,
    Optional,
    Union,
    Mapping,
)

from .config import PackerConfig, SliceConfig
import dataclasses
import numpy as np
import xarray as xr
import pandas as pd
import tensorflow as tf


def _feature_dims(data: xr.Dataset, sample_dims: Sequence[str]) -> Sequence[str]:
    return [str(dim) for dim in data.dims.keys() if dim not in sample_dims]


def _unique_dim_name(
    data: xr.Dataset,
    sample_dims: Sequence[str],
    feature_dim_name_2d_var: str = "feature",
) -> str:
    feature_dims = _feature_dims(data, sample_dims=sample_dims)
    if len(feature_dims) > 0:
        feature_dim_name = "_".join(["feature"] + list(feature_dims))
    else:
        feature_dim_name = feature_dim_name_2d_var
    if feature_dim_name in sample_dims:
        raise ValueError(
            f"The feature dim name ({feature_dim_name}) cannot be one of the "
            f"non-feature dims ({sample_dims})"
        )
    return feature_dim_name


@dataclasses.dataclass
class PackingInfo:
    names: Sequence[str]
    features: Sequence[int]

    @property
    def multi_index(self) -> pd.MultiIndex:
        entries = []
        for name, n_features in zip(self.names, self.features):
            for i in range(n_features):
                entries.append((name, i))
        feature_index = pd.MultiIndex.from_tuples(
            tuple(entries), names=("variable", "z")
        )
        return feature_index


@curry
def clip_sample(
    config: Mapping[str, SliceConfig], data: Mapping[str, tf.Tensor],
) -> Mapping[str, tf.Tensor]:
    return {
        key: value[..., config.get(key, SliceConfig()).slice]
        for (key, value) in data.items()
    }


def pack_tfdataset(
    data: tf.data.Dataset, variable_names: Sequence[str],
) -> Tuple[tf.data.Dataset, PackingInfo]:
    sample = next(iter(data))
    features = []
    for name in variable_names:
        if len(sample[name].shape) > 0:
            features.append(sample[name].shape[-1])
        else:
            features.append(1)
    packing_info = PackingInfo(names=variable_names, features=features)
    return (
        data.map(
            lambda x: tf.concat(list(x[name] for name in variable_names), axis=-1)
        ),
        packing_info,
    )


def pack(
    data: xr.Dataset, sample_dims: Sequence[str], config: Optional[PackerConfig] = None,
) -> Tuple[np.ndarray, pd.MultiIndex]:
    """
    Pack a dataset into a numpy array.

    Args:
        data: dataset to pack
        sample_dims: names of non-feature dimensions, must be present
            in every variable in `data`
    """
    if config is None:
        config = PackerConfig({})
    feature_dim_name = _unique_dim_name(data, sample_dims=sample_dims)

    data_clipped = xr.Dataset(clip(data, config.clip))
    stacked = data_clipped.to_stacked_array(feature_dim_name, sample_dims=sample_dims)
    stacked = stacked.dropna(feature_dim_name)
    return (
        stacked.transpose(*sample_dims, feature_dim_name).values,
        stacked.indexes[feature_dim_name],
    )


def unpack(
    data: np.ndarray, sample_dims: Sequence[str], feature_index: pd.MultiIndex,
) -> xr.Dataset:
    if len(data.shape) == len(sample_dims):
        selection: List[Union[slice, None]] = [slice(None, None) for _ in sample_dims]
        selection = selection + [None]
        data = data[selection]
    da = xr.DataArray(
        data, dims=list(sample_dims) + ["feature"], coords={"feature": feature_index}
    )
    return da.to_unstacked_dataset("feature")


def clip(
    data: Union[xr.Dataset, Mapping[Hashable, xr.DataArray]],
    config: Mapping[Hashable, SliceConfig],
) -> Mapping[Hashable, xr.DataArray]:
    clipped_data = {}
    for variable in data:
        da = data[variable]
        da = _fill_empty_coords(da)
        if variable in config:
            dim = data[variable].dims[-1]
            da = da.isel({dim: config[variable].slice})
        clipped_data[variable] = da
    return clipped_data


def _fill_empty_coords(da):
    # need coord to allow proper unpacking if dim is clipped to
    # different length for different variables. Also needs to be filled
    # for non-clipped variables that have that dim.
    for dim in da.dims:
        if dim not in da.coords:
            da = da.assign_coords({dim: range(da.sizes[dim])})
    return da


def count_features(index: pd.MultiIndex, variable_dim="variable") -> Dict[str, int]:
    variable_idx = index.names.index(variable_dim)
    count: Dict[str, int] = collections.defaultdict(int)
    for item in index:
        variable = item[variable_idx]
        count[variable] += 1
    return count
