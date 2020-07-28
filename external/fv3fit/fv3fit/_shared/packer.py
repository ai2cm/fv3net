from typing import Iterable, TextIO, List, Dict, Tuple, cast
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

    def __init__(self, sample_dim_name, names: Iterable[str]):
        """Initialize the ArrayPacker.

        Args:
            names: variable names to pack.
        """
        self._names = list(names)
        self._n_features: Dict[str, int] = {}
        self._sample_dim_name = sample_dim_name
        self._dims: Dict[str, Iterable[str]] = {}

    @property
    def names(self) -> List[str]:
        """variable names being packed"""
        return self._names

    @property
    def sample_dim_name(self) -> str:
        """name of sample dimension"""
        return self._sample_dim_name

    @property
    def _total_features(self):
        return sum(self._n_features[name] for name in self._names)

    def to_array(self, dataset: xr.Dataset) -> np.ndarray:
        if len(self._n_features) == 0:
            self._n_features = count_features(
                self.names, dataset, self._sample_dim_name
            )
            for name in self.names:
                self._dims[name] = cast(Tuple[str], dataset[name].dims)
        n_samples = dataset.dims[self.sample_dim_name]
        array = np.empty([n_samples, self._total_features])
        i_start = 0
        for name in self.names:
            n_features = self._n_features[name]
            if n_features > 1:
                array[:, i_start : i_start + n_features] = dataset[name]
            else:
                array[:, i_start] = dataset[name]
            i_start += n_features
        return array

    def to_dataset(self, array: np.ndarray) -> xr.Dataset:
        if len(self._n_features) == 0:
            raise RuntimeError(
                "must pack at least once before unpacking, "
                "so dimension lengths are known"
            )
        data_vars = {}
        i_start = 0
        for name in self.names:
            n_features = self._n_features[name]
            if n_features > 1:
                data_vars[name] = (
                    self._dims[name],
                    array[:, i_start : i_start + n_features],
                )
            else:
                data_vars[name] = (self._dims[name], array[:, i_start])
            i_start += n_features
        return xr.Dataset(data_vars)  # type: ignore

    def dump(self, f: TextIO):
        return yaml.safe_dump(
            {
                "n_features": self._n_features,
                "names": self._names,
                "sample_dim_name": self._sample_dim_name,
                "dims": self._dims,
            },
            f,
        )

    @classmethod
    def load(cls, f: TextIO):
        data = yaml.safe_load(f.read())
        packer = cls(data["sample_dim_name"], data["names"])
        packer._n_features = data["n_features"]
        packer._dims = data["dims"]
        return packer


def count_features(names, dataset, sample_dim_name):
    return_dict = {}
    for name in names:
        value = dataset[name]
        if len(value.dims) == 1 and value.dims[0] == sample_dim_name:
            return_dict[name] = 1
        elif len(value.dims) != 2:
            raise ValueError(
                "on first pack can only pack 2D variables, recieved value "
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
