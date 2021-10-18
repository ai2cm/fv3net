import collections
from typing import (
    Hashable,
    Iterable,
    TextIO,
    List,
    Dict,
    Tuple,
    Sequence,
    Optional,
)
import numpy as np
import xarray as xr
import pandas as pd
import yaml


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


def pack(data: xr.Dataset, sample_dim: str) -> Tuple[np.ndarray, pd.MultiIndex]:
    feature_dim_name = _unique_dim_name(data, sample_dim)
    stacked = data.to_stacked_array(feature_dim_name, sample_dims=[sample_dim])
    return (
        stacked.transpose(sample_dim, feature_dim_name).values,
        stacked.indexes[feature_dim_name],
    )


def unpack(
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


class ArrayPacker:
    """
    A class to handle converting xarray datasets to and from numpy arrays.

    Used for ML training/prediction.
    """

    def __init__(self, sample_dim_name, pack_names: Iterable[Hashable]):
        """Initialize the ArrayPacker.

        Args:
            sample_dim_name: dimension name to treat as the sample dimension
            pack_names: variable pack_names to pack
        """
        self._pack_names: List[str] = list(str(s) for s in pack_names)
        self._n_features: Dict[str, int] = {}
        self._sample_dim_name = sample_dim_name
        self._feature_index: Optional[pd.MultiIndex] = None

    @property
    def pack_names(self) -> List[str]:
        """variable pack_names being packed"""
        return self._pack_names

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

    def to_array(self, dataset: xr.Dataset) -> np.ndarray:
        """Convert dataset into a 2D array with [sample, feature] dimensions.

        Variable names inserted into the array are passed on initialization of this
        object. Each of those variables in the dataset must have the sample
        dimension name indicated when this object was initialized, and at most one
        more dimension, considered the feature dimension.

        The first time this is called, the length of the feature dimension for each
        variable is stored, and can be retrieved on `packer.feature_counts`.
        
        Args:
            dataset: dataset containing variables in self.pack_names to pack

        Returns:
            array: 2D [sample, feature] array with data from the dataset
        """
        array, feature_index = pack(dataset[self._pack_names], self._sample_dim_name)
        self._n_features = _count_features(feature_index)
        self._feature_index = feature_index
        return array

    def to_dataset(self, array: np.ndarray) -> xr.Dataset:
        """Restore a dataset from a 2D [sample, feature] array.

        Can only be called after `to_array` is called.

        Args:
            array: 2D [sample, feature] array

        Returns:
            dataset: xarray dataset with data from the given array
        """
        if len(array.shape) > 2:
            raise NotImplementedError("can only restore 2D arrays to datasets")
        if len(self._n_features) == 0 or self._feature_index is None:
            raise RuntimeError(
                "must pack at least once before unpacking, "
                "so dimension lengths are known"
            )
        return unpack(array, self._sample_dim_name, self._feature_index)

    def dump(self, f: TextIO):
        return yaml.safe_dump(
            {
                "pack_names": self._pack_names,
                "sample_dim_name": self._sample_dim_name,
                "feature_index": multiindex_to_tuple(self._feature_index),
            },
            f,
        )

    @classmethod
    def load(cls, f: TextIO):
        data = yaml.safe_load(f.read())
        packer = cls(data["sample_dim_name"], data["pack_names"])
        packer._feature_index = tuple_to_multiindex(data["feature_index"])
        packer._n_features = _count_features(packer._feature_index)
        return packer


def _count_features(index: pd.MultiIndex, variable_dim="variable") -> Dict[str, int]:
    variable_idx = index.names.index(variable_dim)
    count: Dict[str, int] = collections.defaultdict(int)
    for item in index:
        variable = item[variable_idx]
        count[variable] += 1
    return count


def unpack_matrix(
    x_packer: ArrayPacker, y_packer: ArrayPacker, matrix: np.ndarray
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
