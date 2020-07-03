from typing import Tuple
import loaders
import numpy as np
import xarray as xr
from ..sklearn.wrapper import _pack, _unpack
import yaml
import pandas as pd

__all__ = ["ArrayPacker"]


class ArrayPacker:
    def __init__(self, names):
        self._indices = None
        self._names = names

    @property
    def names(self):
        return self._names

    def pack(self, dataset: xr.Dataset) -> np.ndarray:
        packed, indices = pack(dataset[self.names])
        if self._indices is None:
            self._indices = indices
        return packed

    def unpack(self, array: np.ndarray) -> xr.Dataset:
        if self._indices is None:
            raise RuntimeError(
                "must pack at least once before unpacking, "
                "so dimension lengths are known"
            )
        return unpack(array, self._indices)

    def dump(self, f):
        return yaml.safe_dump(
            {
                "indices": multiindex_to_serializable(self._indices),
                "names": self._names,
            },
            f,
        )

    @classmethod
    def load(cls, f):
        data = yaml.safe_load(f.read())
        packer = cls(data["names"])
        packer._indices = multiindex_from_serializable(data["indices"])
        return packer


def multiindex_to_serializable(multiindex):
    return {
        "tuples": tuple(multiindex.to_native_types()),
        "names": tuple(multiindex.names),
    }


def multiindex_from_serializable(data):
    return pd.MultiIndex.from_tuples(
        [[name, int(value)] for name, value in data["tuples"]], names=data["names"]
    )


def pack(dataset) -> Tuple[np.ndarray, np.ndarray]:
    return _pack(dataset, loaders.SAMPLE_DIM_NAME)


def unpack(dataset, feature_indices):
    return _unpack(dataset, loaders.SAMPLE_DIM_NAME, feature_indices)
