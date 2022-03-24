import logging
import numpy as np
import tensorflow as tf
import xarray as xr
from toolz.functoolz import compose_left
from typing import Callable, Mapping, Optional, Sequence

from fv3fit.tfdataset import seq_to_tfdataset
from .transforms import open_netcdf_dataset
from .io import get_nc_files


logger = logging.getLogger(__name__)


def nc_files_to_tf_dataset(
    files: Sequence[str], convert: Callable[[xr.Dataset], Mapping[str, tf.Tensor]],
):

    """
    Convert a list of netCDF paths into a tensorflow dataset.

    Args:
        files: List of local or remote file paths to include in dataset.
            Expected to be 2D ([sample, feature]) or 1D ([sample]) dimensions.
        convert: function to convert netcdf files to tensor dictionaries
    """

    transform = compose_left(*[open_netcdf_dataset, convert])
    return seq_to_tfdataset(files, transform).unbatch().prefetch(tf.data.AUTOTUNE)


def nc_dir_to_tfdataset(
    nc_dir: str,
    convert: Callable[[xr.Dataset], Mapping[str, tf.Tensor]],
    nfiles: Optional[int] = None,
    shuffle: bool = False,
    random_state: Optional[np.random.RandomState] = None,
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

    files = get_nc_files(nc_dir)

    if shuffle:
        if random_state is None:
            random_state = np.random.RandomState(np.random.get_state()[1][0])

        files = random_state.choice(files, size=len(files), replace=False)

    if nfiles is not None:
        files = files[:nfiles]

    return nc_files_to_tf_dataset(files, convert)
