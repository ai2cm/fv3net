import tensorflow as tf
import vcm
import xarray as xr


def get_data(ds, variables) -> tf.Tensor:
    def convert(array: xr.DataArray):
        return tf.convert_to_tensor(array, dtype=tf.float32)

    return tuple([convert(ds[v]) for v in variables])


def read_image_from_url(fs, url, variables):
    sig = (tf.float32,) * len(variables)

    outputs = tf.py_function(lambda url: open_url(fs, url, variables), [url], sig)
    return dict(zip(variables, outputs))


def open_url(fs, url, variables):
    url = url.numpy().decode()
    print(f"opening {url}")
    return get_data(vcm.open_remote_nc(fs, url), variables)


def netcdf_url_to_dataset(url, variables, shuffle=False):
    fs = vcm.get_fs(url)
    d = tf.data.Dataset.from_tensor_slices(sorted(fs.ls(url)))
    if shuffle:
        d = d.shuffle(100_000)
    return d.map(lambda url: read_image_from_url(fs, url, variables))


def load_samples(train_dataset, n_train):
    n_train = 50_000
    train_data = train_dataset.take(n_train).shuffle(n_train).batch(n_train)
    return next(iter(train_data))
