import logging
import numpy as np
import tensorflow as tf
import xarray as xr
from toolz.functoolz import compose_left
from typing import Callable, Optional, Sequence

from .transforms import open_netcdf_dataset
from .io import get_nc_files


logger = logging.getLogger(__name__)


def _seq_to_tf_dataset(source: Sequence, transform: Callable,) -> tf.data.Dataset:
    """
    A general function to convert from a sequence into a tensorflow dataset
    to be used for ML model training.

    Args:
        source: A sequence of data items to be included in the
            dataset.
        transform: function to process data items into a tensor-compatible
            result
    """

    def get_generator():
        for batch in source:
            output = transform(batch)
            yield tf.data.Dataset.from_tensor_slices(output)

    peeked = next(get_generator())
    signature = tf.data.DatasetSpec.from_value(peeked)
    tf_ds = tf.data.Dataset.from_generator(get_generator, output_signature=signature)

    # Flat map goes from generating tf_dataset -> generating tensors
    tf_ds = tf_ds.flat_map(lambda x: x)

    return tf_ds


def nc_files_to_tf_dataset(
    files: Sequence[str], convert: Callable[[xr.Dataset], tf.Tensor],
):

    """
    Convert a list of netCDF paths into a tensorflow dataset.

    Args:
        files: List of local or remote file paths to include in dataset.
            Expected to be 2D ([sample, feature]) or 1D ([sample]) dimensions.
        config: Data preprocessing options for going from xr.Dataset to
            X, y tensor tuples grouped by variable.
    """

    transform = compose_left(*[open_netcdf_dataset, convert])
    return _seq_to_tf_dataset(files, transform)


def nc_dir_to_tf_dataset(
    nc_dir: str,
    convert: Callable[[xr.Dataset], tf.Tensor],
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
