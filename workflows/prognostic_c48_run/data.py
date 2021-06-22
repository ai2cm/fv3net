import tensorflow as tf
import vcm
import xarray as xr
from runtime.emulator import DELP


def get_data(ds, timestep, input_variables):
    def convert(array: xr.DataArray):
        return tf.convert_to_tensor(array, dtype=tf.float32)

    inputs_ = [convert(ds[v]) for v in input_variables]
    outputs = [
        ds[field] + timestep * ds[f"tendency_of_{field}_due_to_fv3_physics"]
        for field in input_variables[:4]
    ] + [ds[DELP]]
    output_tensors = [convert(array) for array in outputs]
    return inputs_ + output_tensors


def read_image_from_url(fs, url, timestep, input_variables):
    input_sig = (tf.float32,) * len(input_variables)
    output_sig = (tf.float32,) * 5

    outputs = tf.py_function(
        lambda url: open_url(fs, url, timestep, input_variables),
        [url],
        # py_function does not support nested output types
        # https://stackoverflow.com/questions/60290340/tensorflow-py-function-nested-output-type
        input_sig + output_sig,
    )
    return (
        tuple(outputs[: len(input_variables)]),
        tuple(outputs[len(input_variables) :]),
    )


def open_url(fs, url, timestep, input_variables):
    url = url.numpy().decode()
    print(f"opening {url}")
    return get_data(vcm.open_remote_nc(fs, url), timestep, input_variables)


def netcdf_url_to_dataset(url, timestep, input_variables, shuffle=False):
    fs = vcm.get_fs(url)
    d = tf.data.Dataset.from_tensor_slices(sorted(fs.ls(url)))
    if shuffle:
        d = d.shuffle(100_000)
    return d.map(lambda url: read_image_from_url(fs, url, timestep, input_variables))


def load_samples(train_dataset, n_train):
    n_train = 50_000
    train_data = train_dataset.take(n_train).shuffle(n_train).batch(n_train)
    return next(iter(train_data))
