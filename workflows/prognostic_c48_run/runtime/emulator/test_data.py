import pytest
import tensorflow as tf
from runtime.emulator.data import netcdf_url_to_dataset, read_image_from_url


@pytest.mark.xfail
def test_read_image_from_url():
    urls = tf.data.Dataset.from_tensor_slices(["not a real url"])
    # these data are not actually loaded
    input_variables = ["u", "v", "t", "q"]
    (u, v, t, q), (uo, vo, to, qo) = urls.map(
        lambda url: read_image_from_url(None, url, None, input_variables)
    )


def test_netcdf_url_to_dataset():
    d = netcdf_url_to_dataset("data/training", variables=["specific_humidity"])
    m = next(iter(d))
    assert isinstance(m["specific_humidity"], tf.Tensor)
