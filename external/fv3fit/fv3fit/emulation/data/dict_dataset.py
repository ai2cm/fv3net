import logging
from typing import Optional, Sequence
import tensorflow as tf

import vcm
from fv3fit.emulation.data.config import Pipeline
from fv3fit.emulation.data.io import get_nc_files
from .transforms import open_netcdf_dataset

__all__ = ["netcdf_url_to_dataset"]
logger = logging.getLogger(__name__)


def read_variables_as_tfdataset(url, variables, transform):
    sig = (tf.float32,) * len(variables)
    # tf.py_function can only wrap functions which output tuples of tensors, not
    # dicts
    outputs = tf.py_function(
        lambda url: read_variables_greedily_as_tuple(url, variables, transform),
        [url],
        sig,
    )

    d = dict(zip(variables, outputs))
    return tf.data.Dataset.from_tensor_slices(d, num_parallel_calls=tf.data.AUTOTUNE)


def read_variables_greedily_as_tuple(url, variables, transform):
    url = url.numpy().decode()
    logger.debug(f"opening {url}")
    ds = open_netcdf_dataset(url)
    ds = transform(ds)
    return tuple([tf.convert_to_tensor(ds[v], dtype=tf.float32) for v in variables])


def netcdf_url_to_dataset(
    url: str,
    variables: Sequence[str],
    transform: Pipeline,
    shuffle: bool = False,
    nfiles: Optional[int] = None,
) -> tf.data.Dataset:
    """Open a url of netcdfs as a tf.data.Dataset of dicts

    Args:
        url: points to a directory of netcdf files.
        variables: a sequence of variable names to load from each netcdf file
        shuffle: if True, shuffle order the netcdf files will be loaded in. Does
            not shuffle BETWEEN files. Reshuffles each epoch
        nfiles: number of files to include in dataset

    Returns:
        a  tensorflow dataset containing dictionaries of tensors. This
        dictionary contains all the variables specified in ``variables``.
    """
    fs = vcm.get_fs(url)
    files = get_nc_files(url, fs)

    if nfiles is not None:
        files = files[:nfiles]

    d = tf.data.Dataset.from_tensor_slices(sorted(files))

    if shuffle:
        d = d.shuffle(len(files))

    return d.interleave(
        lambda url: read_variables_as_tfdataset(url, variables, transform)
    )


def load_samples(train_dataset, n_train):
    train_data = train_dataset.take(n_train).shuffle(n_train).batch(n_train)
    return next(iter(train_data))
