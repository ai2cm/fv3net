from typing import Sequence
import tensorflow as tf
import vcm
from fv3fit.emulation.data.io import get_nc_files

__all__ = ["netcdf_url_to_dataset"]


def read_variables_as_dict(fs, url, variables):
    sig = (tf.float32,) * len(variables)
    outputs = tf.py_function(
        lambda url: read_variables_greedily_as_tuple(fs, url, variables), [url], sig
    )
    return dict(zip(variables, outputs))


def read_variables_greedily_as_tuple(fs, url, variables):
    url = url.numpy().decode()
    print(f"opening {url}")
    ds = vcm.open_remote_nc(fs, url)
    return tuple([tf.convert_to_tensor(ds[v], dtype=tf.float32) for v in variables])


def netcdf_url_to_dataset(
    url: str, variables: Sequence[str], shuffle: bool = False
) -> tf.data.Dataset:
    """Open a url of netcdfs as a tf.data.Dataset of dicts

    Args:
        url: points to a directory of netcdf files.
        variables: a sequence of variable names to load from each netcdf file
        shuffle: if True, shuffle order the netcdf files will be loaded in. Does
            not shuffle BETWEEN files.
    
    Returns:
        a  tensorflow dataset containing dictionaries of tensors. This
        dictionary contains all the variables specified in ``variables``.
    """
    fs = vcm.get_fs(url)
    files = get_nc_files(url, fs)
    d = tf.data.Dataset.from_tensor_slices(sorted(files))
    if shuffle:
        d = d.shuffle(100_000)
    return d.map(lambda url: read_variables_as_dict(fs, url, variables))


def load_samples(train_dataset, n_train):
    train_data = train_dataset.take(n_train).shuffle(n_train).batch(n_train)
    return next(iter(train_data))
