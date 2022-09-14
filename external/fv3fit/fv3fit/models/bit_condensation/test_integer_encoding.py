import tensorflow as tf
import pytest
from fv3fit.models.bit_condensation import integer_encoding


def test_IntEncoder_encode():
    dt = tf.int16
    encoder = integer_encoding.IntEncoder(1.0, dt=dt)
    x = tf.constant(0.5)
    y = encoder.encode(x)
    assert (dt.size * 8,) == tuple(y.shape)


def test_IntEncoder_decode():
    encoder = integer_encoding.IntEncoder(1.0, dt=tf.int32)
    x = tf.constant(0.5)
    y = encoder.encode(x)
    x_decoded = encoder.decode(y)
    assert x_decoded.numpy() == pytest.approx(0.5)


@pytest.mark.parametrize("cls", [integer_encoding.Log, integer_encoding.IEEE])
def test_Log_encode(cls):
    coder = cls()
    a = -0.5
    x = tf.constant(a)
    y = coder.encode(x)
    x_decoded = coder.decode(y)
    assert x_decoded.numpy() == pytest.approx(a)
