import tensorflow as tf
from math import pi
from fv3fit.emulation.layers import StableDynamics, StableEncoder
import pytest


def test_complex_multiplication():
    out = tf.complex(0.0, pi / 2)

    y = tf.exp(out)
    assert pytest.approx(0.0, abs=1e-7) == tf.math.real(y).numpy()
    assert pytest.approx(1.0, abs=1e-7) == tf.math.imag(y).numpy()


def to_float(x):
    return x.numpy().item()


def test_StableDynamics_is_contracting():
    z = tf.ones([1, 10], dtype=tf.dtypes.complex64)
    dyn = StableDynamics(10)

    def norm(z):
        return to_float(tf.math.real(tf.reduce_sum(tf.math.conj(z) * z)))

    assert norm(dyn(z)) <= norm(z)


def test_StableEncoder_encoder_loss_has_reasonable_magnitude():
    n = 23
    x = tf.ones([n, 10])
    auxiliary = tf.ones([n, 3])

    encoder = StableEncoder()
    encoder([x, auxiliary])
    assert 0.1 < sum(encoder.losses) < 10


def test_StableEncoder_prediction_has_same_size():
    n = 23
    x = tf.ones([n, 10])
    auxiliary = tf.ones([n, 3])

    encoder = StableEncoder()
    y = encoder([x, auxiliary])
    assert x.shape == y.shape
