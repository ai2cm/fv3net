import logging
import tensorflow as tf
import xarray as xr
from toolz.functoolz import compose_left
from typing import Callable, Sequence

from .config import TransformConfig
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
    tf_ds = tf_ds.prefetch(tf.data.AUTOTUNE).flat_map(lambda x: x)

    return tf_ds


def nc_files_to_tf_dataset(files: Sequence[str], config: TransformConfig):

    """
    Convert a list of netCDF paths into a tensorflow dataset.

    Args:
        files: List of local or remote file paths to include in dataset.
            Expected to be 2D ([sample, feature]) or 1D ([sample]) dimensions.
        config: Data preprocessing options for going from xr.Dataset to
            X, y tensor tuples grouped by variable.
    """

    transform = compose_left(*[open_netcdf_dataset, config])

    return _seq_to_tf_dataset(files, transform)


def nc_dir_to_tf_dataset(nc_dir: str, config: TransformConfig):

    """
    Convert a directory of netCDF files into a tensorflow dataset.

    Args:
        nc_dir: Path to a directory of netCDFs to include in dataset.
            Expected to be 2D ([sample, feature]) or 1D ([sample]) dimensions.
        config: Data preprocessing options for going from xr.Dataset to
            X, y tensor tuples grouped by variable.
    """

    files = get_nc_files(nc_dir)
    return nc_files_to_tf_dataset(files, config)


def batches_to_tf_dataset(batches: Sequence[xr.Dataset], config: TransformConfig):

    """
    Convert a batched data sequence of datasets into a tensorflow dataset

    Args:
        batches: A batched data sequence (e.g., from loaders.batches) with
            dimensions 2D ([sample, feature]) or 1D ([sample]) dimensions.
        config: Data preprocessing options for going from xr.Dataset to
            X, y tensor tuples grouped by variable.
    """

    return _seq_to_tf_dataset(batches, config)
