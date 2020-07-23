from typing import Iterable, TextIO, List
import numpy as np
import xarray as xr
import yaml
import pandas as pd


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
        self._indices = None
        self._names = list(names)
        self._sample_dim_name = sample_dim_name

    @property
    def names(self) -> List[str]:
        """variable names being packed"""
        return self._names

    @property
    def sample_dim_name(self) -> str:
        """name of sample dimension"""
        return self._sample_dim_name

    def to_array(self, dataset: xr.Dataset) -> np.ndarray:
        packed, indices = pack(
            dataset[self.names], self._sample_dim_name
        )  # type: ignore
        if self._indices is None:
            self._indices = indices
        return packed

    def to_dataset(self, array: np.ndarray) -> xr.Dataset:
        if self._indices is None:
            raise RuntimeError(
                "must pack at least once before unpacking, "
                "so dimension lengths are known"
            )
        return unpack(array, self._sample_dim_name, self._indices)

    def dump(self, f: TextIO):
        return yaml.safe_dump(
            {
                "indices": _multiindex_to_serializable(self._indices),
                "names": self._names,
                "sample_dim_name": self._sample_dim_name,
            },
            f,
        )

    @classmethod
    def load(cls, f: TextIO):
        data = yaml.safe_load(f.read())
        packer = cls(data["sample_dim_name"], data["names"])
        packer._indices = _multiindex_from_serializable(data["indices"])
        return packer


def _multiindex_to_serializable(multiindex: pd.MultiIndex) -> dict:
    """Convert a pandas MultiIndex into something that can be serialized to yaml"""
    return {
        "tuples": tuple(multiindex.to_native_types()),
        "names": tuple(multiindex.names),
    }


def _multiindex_from_serializable(data: dict) -> pd.MultiIndex:
    """Convert serializable yaml to a pandas MultiIndex"""
    return pd.MultiIndex.from_tuples(
        [[name, int(value)] for name, value in data["tuples"]], names=data["names"]
    )
