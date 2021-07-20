import numpy as np
import tensorflow as tf
from runtime.emulator.layers import (
    NormLayer,
    UnNormLayer,
    ScalarNormLayer,
)
from hypothesis import given
from hypothesis.strategies import integers

def test_NormLayer():
    u = tf.Variable([[0.0], [1.0]], dtype=tf.float32)
    layer = NormLayer()
    layer.fit(u)
    norm = layer(u)
    expected = np.array([[-1.0], [1.0]])
    np.testing.assert_allclose(expected, norm, rtol=1e-6)


def test_NormLayer_no_trainable_variables():
    u = tf.Variable([[0.0], [1.0]], dtype=tf.float32)
    layer = NormLayer()
    layer(u)

    assert [] == layer.trainable_variables


def test_NormLayer_gradient_works():
    u = tf.Variable([[0.0, 0.0], [1.0, 2.0]], dtype=tf.float32)
    layer = NormLayer()
    layer(u)

    with tf.GradientTape() as tape:
        y = layer(u)
    (g,) = tape.gradient(y, [u])
    expected = 1 / (layer.sigma + layer.epsilon)
    np.testing.assert_array_almost_equal(expected, g[0, :])


@given(integers())
def test_ScalarNormLayer(seed):
    tf.random.set_seed(seed)
    u = tf.random.uniform([2, 3])
    norm = NormLayer()
    norm.fit(u)

    unnorm = UnNormLayer()
    unnorm.fit(u)
    round_tripped = unnorm(norm(u))
    np.testing.assert_almost_equal(u.numpy(), round_tripped.numpy())


def test_scalar_norm_layer():
    input = np.array([[1, 2], [-1, -2]], dtype=np.float32)
    expected = input * np.sqrt(10 / 4)

    norm = ScalarNormLayer()
    norm.fit(input)

    np.testing.assert_allclose(norm(input).numpy(), expected)
