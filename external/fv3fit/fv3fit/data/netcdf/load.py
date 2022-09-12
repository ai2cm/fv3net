import contextlib
import logging
from dataclasses import dataclass
from typing import Callable, Iterator, Mapping, Optional, Sequence

import numpy as np
import tensorflow as tf
import xarray as xr
from fv3fit.data.base import TFDatasetLoader, register_tfdataset_loader
from fv3fit.tfdataset import iterable_to_tfdataset
from toolz.functoolz import compose_left

from .io import CACHE_DIR, download_cached, get_nc_files

logger = logging.getLogger(__name__)

__all__ = ["NetcdfDirLoader", "nc_dir_to_tfdataset"]


@contextlib.contextmanager
def open_dataset(path: str) -> Iterator[xr.Dataset]:
    ds = xr.open_dataset(path)
    yield ds
    ds.close()


def open_netcdf_dataset(path: str, cache=None) -> xr.Dataset:
    """Open a netcdf from a local/remote path"""
    local_path = download_cached(path, cache)
    with open_dataset(local_path) as ds:
        return ds.load()


def nc_files_to_tf_dataset(
    files: Sequence[str],
    convert: Callable[[xr.Dataset], Mapping[str, tf.Tensor]],
    cache: str = CACHE_DIR,
):

    """
    Convert a list of netCDF paths into a tensorflow dataset.

    Args:
        files: List of local or remote file paths to include in dataset.
            Expected to be 2D ([sample, feature]) or 1D ([sample]) dimensions.
        convert: function to convert netcdf files to tensor dictionaries
    """

    transform = compose_left(lambda path: open_netcdf_dataset(path, cache), convert)
    return iterable_to_tfdataset(files, transform).prefetch(tf.data.AUTOTUNE)


def nc_dir_to_tfdataset(
    nc_dir: str,
    convert: Callable[[xr.Dataset], Mapping[str, tf.Tensor]],
    nfiles: Optional[int] = None,
    shuffle: bool = False,
    random_state: Optional[np.random.RandomState] = None,
    cache: Optional[str] = None,
) -> tf.data.Dataset:
    """
    Convert a directory of netCDF files into a tensorflow dataset.

    Args:
        nc_dir: Path to a directory of netCDFs to include in dataset.
            Expected to be 2D ([sample, feature]) or 1D ([sample]) dimensions.
        nfiles: Limit to number of files
        shuffle: Randomly order the file ingestion into the dataset
        random_state: numpy random number generator for seeded shuffle
    """
    cache = cache or CACHE_DIR

    files = get_nc_files(nc_dir)

    if shuffle:
        if random_state is None:
            random_state = np.random.RandomState(np.random.get_state()[1][0])

        files = random_state.choice(files, size=len(files), replace=False)

    if nfiles is not None:
        files = files[:nfiles]

    return nc_files_to_tf_dataset(files, convert)


def to_tensor(
    ds: xr.Dataset, variable_names: Sequence[str], dtype=tf.float32
) -> Mapping[str, tf.Tensor]:
    return {key: tf.convert_to_tensor(ds[key], dtype=dtype) for key in variable_names}


@register_tfdataset_loader
@dataclass
class NetcdfDirLoader(TFDatasetLoader):
    """Loads a folder of netCDF files at given path
    """

    url: str
    nfiles: Optional[int] = None
    shuffle: bool = True
    seed: int = 0

    @property
    def dtype(self):
        return tf.float32

    def convert(
        self, ds: xr.Dataset, variables: Sequence[str]
    ) -> Mapping[str, tf.Tensor]:
        return to_tensor(ds, variables, dtype=self.dtype)

    def open_tfdataset(
        self, local_download_path: Optional[str], variable_names: Sequence[str],
    ) -> tf.data.Dataset:
        def convert(x):
            return self.convert(x, variable_names)

        return nc_dir_to_tfdataset(
            self.url,
            convert=convert,
            nfiles=self.nfiles,
            shuffle=self.shuffle,
            random_state=np.random.RandomState(self.seed),
            cache=local_download_path,
        )

    @classmethod
    def from_dict(cls, d: dict) -> "TFDatasetLoader":
        return cls(**d)
