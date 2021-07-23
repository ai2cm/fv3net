from fv3fit.emulation.layers.norm import MaxProfileStdDenormLayer
import pytest
import numpy as np
import tensorflow as tf

from fv3fit.emulation import layers


@pytest.mark.parametrize(
    "norm_cls, denorm_cls, expected",
    [
        (
            layers.StandardNormLayer,
            layers.StandardDenormLayer,
            [[-1.0, -1.0], [1.0, 1.0]]
        ),
        (
            layers.MaxProfileStdNormLayer,
            layers.MaxProfileStdDenormLayer,
            [[-0.5, -1.0], [0.5, 1.0]]
        )

    ]
)
def test_normalize_layers(norm_cls, denorm_cls, expected):
    input_arr = np.array([[0.0, 0.0], [1.0, 2.0]])
    u = tf.Variable(input_arr, dtype=tf.float32)
    norm_layer = norm_cls()
    denorm_layer = denorm_cls()
    norm_layer.fit(u)
    denorm_layer.fit(u)

    norm = norm_layer(u)
    expected = np.array(expected)
    np.testing.assert_allclose(norm, expected, rtol=1e-6, atol=1e-6)
    denorm = denorm_layer(norm)
    np.testing.assert_allclose(denorm, input_arr, rtol=1e-6, atol=1e-6)


def test_StandardNormLayer_no_trainable_variables():
    u = tf.Variable([[0.0], [1.0]], dtype=tf.float32)
    layer = layers.StandardNormLayer()
    layer(u)

    assert [] == layer.trainable_variables


def test_NormLayer_gradient_works():
    u = tf.Variable([[0.0, 0.0], [1.0, 2.0]], dtype=tf.float32)
    layer = layers.StandardNormLayer()
    layer(u)

    with tf.GradientTape() as tape:
        y = layer(u)
    (g,) = tape.gradient(y, [u])
    expected = 1 / (layer.sigma + layer.epsilon)
    np.testing.assert_array_almost_equal(expected, g[0, :])
