import logging
import os
import numpy as np
import tensorflow as tf
import xarray as xr
from toolz import compose_left
from typing import Any, Callable, List, Optional, Sequence, Union

from vcm import open_remote_nc, get_fs


logger = logging.getLogger(__name__)


def open_netcdf(path: str) -> xr.Dataset:

    fs = get_fs(path)
    data = open_remote_nc(fs, path)

    return data


def get_nc_files(path) -> List:

    fs = get_fs(path)
    files = list(fs.glob(os.path.join(path, "*.nc")))

    return files


def _get_generator_constructor(batched_source, processing_func):

    def get_generator():
        for batch in batched_source:
            output = processing_func(batch)
            yield tf.data.Dataset.from_tensor_slices(output)

    return get_generator


def batched_to_tf_dataset(
    batched_source: Sequence,
    processing_func: Callable[[Any], Union[tf.Tensor, np.ndarray]],
) -> tf.data.Dataset:

    get_generator = _get_generator_constructor(batched_source, processing_func)

    peeked = next(get_generator())
    signature = tf.data.DatasetSpec.from_value(peeked)
    tf_ds = tf.data.Dataset.from_generator(
        get_generator,
        output_signature=signature
    )

    # Interleave is required with a generator that yields a tf.dataset
    tf_ds = tf_ds.prefetch(tf.data.AUTOTUNE).interleave(lambda x: x)

    return tf_ds


def nc_dir_to_tf_dataset(path, preprocessing_func, num_parallel_calls=None):

    files: List = get_nc_files(path)
    preproc = compose_left(*[open_netcdf, preprocessing_func])
    return batched_to_tf_dataset(files, preproc, num_parallel_calls=num_parallel_calls)
