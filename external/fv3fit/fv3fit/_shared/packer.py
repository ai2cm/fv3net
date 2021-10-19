import collections
import dataclasses
from typing import (
    Hashable,
    TextIO,
    List,
    Dict,
    Tuple,
    Sequence,
    Optional,
    Mapping,
)
import numpy as np
import xarray as xr
import pandas as pd
import yaml
import vcm


def _feature_dims(data: xr.Dataset, sample_dim: str) -> Sequence[str]:
    return [str(dim) for dim in data.dims.keys() if dim != sample_dim]


def _unique_dim_name(
    data: xr.Dataset, sample_dim: str, feature_dim_name_2d_var: str = "feature"
) -> str:
    feature_dims = _feature_dims(data, sample_dim)
    if len(feature_dims) > 0:
        feature_dim_name = "_".join(["feature"] + list(feature_dims))
    else:
        feature_dim_name = feature_dim_name_2d_var
    if sample_dim == feature_dim_name:
        raise ValueError(
            f"The sample dim name ({sample_dim}) cannot be the same "
            f"as the feature dim name ({feature_dim_name})"
        )
    return feature_dim_name


def _pack(data: xr.Dataset, sample_dim: str) -> Tuple[np.ndarray, pd.MultiIndex]:
    feature_dim_name = _unique_dim_name(data, sample_dim)
    stacked = data.to_stacked_array(feature_dim_name, sample_dims=[sample_dim])
    return (
        stacked.transpose(sample_dim, feature_dim_name).values,
        stacked.indexes[feature_dim_name],
    )


def _unpack(
    data: np.ndarray, sample_dim: str, feature_index: pd.MultiIndex
) -> xr.Dataset:
    if len(data.shape) == 1:
        data = data[:, None]
    da = xr.DataArray(
        data, dims=[sample_dim, "feature"], coords={"feature": feature_index}
    )
    return da.to_unstacked_dataset("feature")


def multiindex_to_tuple(index: pd.MultiIndex) -> tuple:
    return list(index.names), list(index.to_list())


def tuple_to_multiindex(d: tuple) -> pd.MultiIndex:
    names, list_ = d
    return pd.MultiIndex.from_tuples(list_, names=names)


@dataclasses.dataclass
class ContiguousSlice:
    start: Optional[int]
    stop: Optional[int]


ClipConfig = Mapping[Hashable, Mapping[str, ContiguousSlice]]


@dataclasses.dataclass
class PackerConfig:
    pack_names: Optional[Sequence[str]] = None
    clip_indices: ClipConfig = dataclasses.field(default_factory=dict)


class Unpacker:
    """A class to handle converting from numpy arrays to xarray datasets."""

    def __init__(self, sample_dim_name: str, feature_index: pd.MultiIndex):
        """Initialize the Unpacker.

        Args:
            sample_dim_name: name of sample dimension.
            feature_index: index required to restore xr.Dataset from np.ndarray.
        """
        self._sample_dim_name = sample_dim_name
        self._feature_index = feature_index
        self._n_features = _count_features(self._feature_index)

    @property
    def pack_names(self) -> List[str]:
        """variable pack_names being packed"""
        return list(self._n_features.keys())

    @property
    def sample_dim_name(self) -> str:
        """name of sample dimension"""
        return self._sample_dim_name

    @property
    def feature_counts(self) -> dict:
        return self._n_features.copy()

    @property
    def _total_features(self):
        return sum(self._n_features[name] for name in self._pack_names)

    def to_dataset(self, array: np.ndarray) -> xr.Dataset:
        """Restore a dataset from a 2D [sample, feature] array.

        Args:
            array: 2D [sample, feature] array

        Returns:
            dataset: xarray dataset with data from the given array
        """
        return _unpack(array, self._sample_dim_name, self._feature_index)

    def dump(self, f: TextIO):
        return yaml.safe_dump(
            {
                "sample_dim_name": self._sample_dim_name,
                "feature_index": multiindex_to_tuple(self._feature_index),
            },
            f,
        )

    @classmethod
    def load(cls, f: TextIO):
        data = yaml.safe_load(f.read())
        return cls(data["sample_dim_name"], tuple_to_multiindex(data["feature_index"]))


def _count_features(index: pd.MultiIndex, variable_dim="variable") -> Dict[str, int]:
    variable_idx = index.names.index(variable_dim)
    count: Dict[str, int] = collections.defaultdict(int)
    for item in index:
        variable = item[variable_idx]
        count[variable] += 1
    return count


def unpack_matrix(
    x_packer: Unpacker, y_packer: Unpacker, matrix: np.ndarray
) -> xr.Dataset:
    """Unpack a matrix

    Args:
        x_packer: packer for the rows of the matrix
        y_packer: packer for the columns of the matrix
        matrix: the matrix to be unpacked
    Returns:
        a Dataset

    """
    jacobian_dict = {}
    j = 0
    for in_name in x_packer.pack_names:
        i = 0
        for out_name in y_packer.pack_names:
            size_in = x_packer.feature_counts[in_name]
            size_out = y_packer.feature_counts[out_name]

            jacobian_dict[(in_name, out_name)] = xr.DataArray(
                matrix[i : i + size_out, j : j + size_in], dims=[out_name, in_name],
            )
            i += size_out
        j += size_in

    return xr.Dataset(jacobian_dict)  # type: ignore


def pack(data: xr.Dataset, sample_dim: str) -> Tuple[np.ndarray, Unpacker]:
    """Pack an xarray dataset into a numpy array.

    Args:
        data: the data to be packed.
        sample_dim: name of sample dimension. All other dims will be flattened to a
            single 'feature' dimension.
        config: configuration of packing.

    Returns:
        tuple of packed array and Unpacker object.
    """
    array, index = _pack(data, sample_dim)
    return array, Unpacker(sample_dim, index)
