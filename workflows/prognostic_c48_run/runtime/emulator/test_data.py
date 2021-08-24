import tensorflow as tf
from fv3fit.emulation.data import netcdf_url_to_dataset


def test_netcdf_url_to_dataset():
    d = netcdf_url_to_dataset("data/training", variables=["specific_humidity"])
    m = next(iter(d))
    assert isinstance(m["specific_humidity"], tf.Tensor)
