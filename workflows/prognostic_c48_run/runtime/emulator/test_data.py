import pytest
from runtime.emulator import data
import tensorflow as tf


@pytest.mark.xfail
def test_read_image_from_url():
    urls = tf.data.Dataset.from_tensor_slices(["not a real url"])
    # these data are not actually loaded
    input_variables = ["u", "v", "t", "q"]
    (u, v, t, q), (uo, vo, to, qo) = urls.map(
        lambda url: data.read_image_from_url(None, url, None, input_variables)
    )
