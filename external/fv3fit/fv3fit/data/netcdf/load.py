import logging
from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Sequence

from pathlib import Path
import numpy as np
import re
import tensorflow as tf
import xarray as xr
from fv3fit.data.base import TFDatasetLoader, register_tfdataset_loader
from fv3fit.tfdataset import iterable_to_tfdataset
from toolz.functoolz import compose_left

import dacite

from .io import CACHE_DIR, download_cached, get_nc_files

logger = logging.getLogger(__name__)

__all__ = ["NCDirLoader", "nc_dir_to_tfdataset"]


def open_netcdf_dataset(path: str, cache=None) -> xr.Dataset:
    """Open a netcdf from a local/remote path"""
    local_path = download_cached(path, cache)
    ds = xr.open_dataset(local_path)
    try:
        return ds.load()
    finally:
        ds.close()


def nc_files_to_tf_dataset(
    files: Sequence[str],
    convert: Callable[[xr.Dataset], Mapping[str, tf.Tensor]],
    cache: str = CACHE_DIR,
    varying_first_dim: bool = False,
):

    """
    Convert a list of netCDF paths into a tensorflow dataset.

    Args:
        files: List of local or remote file paths to include in dataset.
            Expected to be 2D ([sample, feature]) or 1D ([sample]) dimensions.
        convert: function to convert netcdf files to tensor dictionaries
    """

    transform = compose_left(lambda path: open_netcdf_dataset(path, cache), convert)
    return iterable_to_tfdataset(files, transform, varying_first_dim).prefetch(
        tf.data.AUTOTUNE
    )


def nc_dir_to_tfdataset(
    nc_dir: str,
    convert: Callable[[xr.Dataset], Mapping[str, tf.Tensor]],
    nfiles: Optional[int] = None,
    shuffle: bool = False,
    random_state: Optional[np.random.RandomState] = None,
    cache: Optional[str] = None,
    match: Optional[str] = None,
    varying_first_dim: bool = False,
) -> tf.data.Dataset:
    """
    Convert a directory of netCDF files into a tensorflow dataset.

    Args:
        nc_dir: Path to a directory of netCDFs to include in dataset.
            Expected to be 2D ([sample, feature]) or 1D ([sample]) dimensions.
        nfiles: Limit to number of files
        shuffle: Randomly order the file ingestion into the dataset
        random_state: numpy random number generator for seeded shuffle
        cache: directory to cache datat at. The default is $pwd/.cache.
        match: string to filter filenames via a regexp search
    """
    cache = cache or CACHE_DIR

    files = get_nc_files(nc_dir)

    if match is not None:
        files = [f for f in files if re.search(match, Path(f).name)]

    if shuffle:
        if random_state is None:
            random_state = np.random.RandomState(
                np.random.get_state()[1][0]  # type: ignore
            )

        files = random_state.choice(
            files, size=len(files), replace=False  # type: ignore
        )

    if nfiles is not None:
        files = files[:nfiles]

    return nc_files_to_tf_dataset(files, convert, varying_first_dim=varying_first_dim)


def to_tensor(
    ds: xr.Dataset, variable_names: Sequence[str], dtype=tf.float32
) -> Mapping[str, tf.Tensor]:
    return {key: tf.convert_to_tensor(ds[key], dtype=dtype) for key in variable_names}


@register_tfdataset_loader
@dataclass
class NCDirLoader(TFDatasetLoader):
    """Loads a folder of netCDF files at given path

    Each file must have identical CDL scheme returned by ``ncdump -h``.
    Typically, samples will be defined over the first dimension to allow easy
    use with ``tf.data.Dataset.unbatch``. See this example::

        netcdf A {
            dimensions:
                // the dimensions can be named anything
                // but the first dimension must be the same size
                // and should correspond to "samples".
                sample = 10;
                z = 5;
            variables:
                // all arrays share the first dimension
                float a(sample, z);
                float b(sample);
            data:
                a = 0, 1, 2, 3;
        }

    Attributes:
        url: a path to the directory
        nfiles: the number of files to load
        shuffle: Whether the files are opened in a shuffled order
        seed: The random seed used for filename shuffling.
        dim_order: Order of dimensions in tensors. Will expand missing
            dims into size 1 dim if not in a data array.
        varying_first_dim: If true, allow number of samples to be different
            across batches.

    """

    url: str
    nfiles: Optional[int] = None
    shuffle: bool = True
    seed: int = 0
    dim_order: Optional[Sequence[str]] = None
    varying_first_dim: bool = False

    @property
    def dtype(self):
        return tf.float32

    def convert(
        self, ds: xr.Dataset, variables: Sequence[str]
    ) -> Mapping[str, tf.Tensor]:
        if self.dim_order:
            # Operates on individual data arrays to allow for inserting z dim
            # of size 1 for 2D variables. This gives all tensors the same number
            # of dimensions for easier concatenation later on in training.
            tensors = {}
            for key in variables:
                data_array = self._ensure_consistent_dims(ds[key])
                tensors[key] = tf.convert_to_tensor(data_array, dtype=self.dtype)
            return tensors
        else:
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
            varying_first_dim=self.varying_first_dim,
        )

    @classmethod
    def from_dict(cls, d: dict) -> "TFDatasetLoader":
        return dacite.from_dict(cls, d, config=dacite.Config(strict=True))

    def _ensure_consistent_dims(self, data_array: xr.DataArray):
        if self.dim_order:
            extra_dims_in_data_array = set(data_array.dims) - set(self.dim_order)
            missing_dims_in_data_array = set(self.dim_order) - set(data_array.dims)
            if len(extra_dims_in_data_array) > 0:
                raise ValueError(
                    f"Extra dimensions {extra_dims_in_data_array} in data that are not "
                    f"included in configured dimension order {self.dim_order}."
                    "Make sure these are included in the configuration dim_order."
                )
            da = data_array
            for missing_dim in missing_dims_in_data_array:
                da = data_array.expand_dims(dim=missing_dim)
            da = da.transpose(*self.dim_order)
        return da
