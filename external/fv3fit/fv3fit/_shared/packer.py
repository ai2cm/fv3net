from typing import Iterable, TextIO, List, Dict, Tuple, cast, Mapping
import numpy as np
import xarray as xr
import yaml


def _unique_dim_name(data):
    return "".join(data.dims)


def pack(data, sample_dim):
    feature_dim_name = _unique_dim_name(data)
    stacked = data.to_stacked_array(feature_dim_name, sample_dims=[sample_dim])
    return (
        stacked.transpose(sample_dim, feature_dim_name).data,
        stacked.indexes[feature_dim_name],
    )


def unpack(data: np.ndarray, sample_dim, feature_index):
    da = xr.DataArray(
        data, dims=[sample_dim, "feature"], coords={"feature": feature_index}
    )
    return da.to_unstacked_dataset("feature")


class ArrayPacker:
    """
    A class to handle converting xarray datasets to and from numpy arrays.

    Used for ML training/prediction.
    """

    def __init__(self, sample_dim_name, pack_names: Iterable[str]):
        """Initialize the ArrayPacker.

        Args:
            sample_dim_name: dimension name to treat as the sample dimension
            pack_names: variable pack_names to pack
        """
        self._pack_names = list(pack_names)
        self._n_features: Dict[str, int] = {}
        self._sample_dim_name = sample_dim_name
        self._dims: Dict[str, Iterable[str]] = {}

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

        On subsequent calls, the feature dimensions are broadcast
        to have this length. This ensures the returned array has the same shape on
        subsequent calls, and allows packing a dataset of scalars against
        [sample, feature] arrays.
        
        Args:
            dataset: dataset containing variables in self.pack_names to pack

        Returns:
            array: 2D [sample, feature] array with data from the dataset
        """
        if len(self._n_features) == 0:
            self._n_features = count_features(
                self.pack_names, dataset, self._sample_dim_name
            )
            for name in self.pack_names:
                self._dims[name] = cast(Tuple[str], dataset[name].dims)
        array = to_array(dataset, self.pack_names, self.feature_counts)
        return array

    def to_dataset(self, array: np.ndarray) -> xr.Dataset:
        """Restore a dataset from a 2D [sample, feature] array.

        Restores dimension names, but does not restore coordinates or attributes.

        Can only be called after `to_array` is called.

        Args:
            array: 2D [sample, feature] array

        Returns:
            dataset: xarray dataset with data from the given array
        """
        if len(self._n_features) == 0:
            raise RuntimeError(
                "must pack at least once before unpacking, "
                "so dimension lengths are known"
            )
        return to_dataset(array, self.pack_names, self._dims, self.feature_counts)

    def dump(self, f: TextIO):
        return yaml.safe_dump(
            {
                "n_features": self._n_features,
                "pack_names": self._pack_names,
                "sample_dim_name": self._sample_dim_name,
                "dims": self._dims,
            },
            f,
        )

    @classmethod
    def load(cls, f: TextIO):
        data = yaml.safe_load(f.read())
        packer = cls(data["sample_dim_name"], data["pack_names"])
        packer._n_features = data["n_features"]
        packer._dims = data["dims"]
        return packer


def to_array(
    dataset: xr.Dataset, pack_names: Iterable[str], feature_counts: Mapping[str, int]
):
    """Convert dataset into a 2D array with [sample, feature] dimensions.

    The first dimension of each variable to pack is assumed to be the sample dimension,
    and the second (if it exists) is assumed to be the feature dimension.
    Each variable must be 1D or 2D.
    
    Args:
        dataset: dataset containing variables in self.pack_names to pack
        pack_names: names of variables to pack
        feature_counts: number of features for each variable

    Returns:
        array: 2D [sample, feature] array with data from the dataset
    """
    # we can assume here that the first dimension is the sample dimension
    n_samples = dataset[pack_names[0]].shape[0]
    total_features = sum(feature_counts[name] for name in pack_names)
    array = np.empty([n_samples, total_features])
    i_start = 0
    for name in pack_names:
        n_features = feature_counts[name]
        if n_features > 1:
            array[:, i_start : i_start + n_features] = dataset[name]
        else:
            array[:, i_start] = dataset[name]
        i_start += n_features
    return array


def to_dataset(
    array: np.ndarray,
    pack_names: Iterable[str],
    dimensions: Mapping[str, Iterable[str]],
    feature_counts: Mapping[str, int],
):
    """Restore a dataset from a 2D [sample, feature] array.

    Restores dimension names, but does not restore coordinates or attributes.

    Can only be called after `to_array` is called.

    Args:
        array: 2D [sample, feature] array
        pack_names: names of variables to unpack
        dimensions: mapping which provides a list of dimensions for each variable
        feature_counts: mapping which provides a number of features for each variable

    Returns:
        dataset: xarray dataset with data from the given array
    """
    data_vars = {}
    i_start = 0
    for name in pack_names:
        n_features = feature_counts[name]
        if n_features > 1:
            data_vars[name] = (
                dimensions[name],
                array[:, i_start : i_start + n_features],
            )
        else:
            data_vars[name] = (dimensions[name], array[:, i_start])
        i_start += n_features
    return xr.Dataset(data_vars)  # type: ignore


def count_features(
    quantity_names: Iterable[str], dataset: xr.Dataset, sample_dim_name: str
) -> Mapping[str, int]:
    """Count the number of ML outputs corresponding to a set of quantities in a dataset.

    The first dimension of all variables indicated must be the sample dimension,
    and they must have at most one other dimension (treated as the "feature" dimension).

    Args:
        quantity_names: names of variables to include in the count
        dataset: a dataset containing the indicated variables
        sample_dim_name: dimension to treat as the "sample" dimension, any other
            dimensions are treated as a "feature" dimension.
    """

    return_dict = {}
    for name in quantity_names:
        value = dataset[name]
        if len(value.dims) == 1 and value.dims[0] == sample_dim_name:
            return_dict[name] = 1
        elif len(value.dims) > 2:
            raise ValueError(
                "can only pack 1D or 2D variables, recieved value "
                f"for {name} with dimensions {value.dims}"
            )
        elif value.dims[0] != sample_dim_name:
            raise ValueError(
                f"cannot pack value for {name} whose first dimension is not the "
                f"sample dimension ({sample_dim_name}), has dims {value.dims}"
            )
        else:
            return_dict[name] = value.shape[1]
    return return_dict
