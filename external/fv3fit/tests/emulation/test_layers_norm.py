import pytest
import numpy as np
import tensorflow as tf

from fv3fit.emulation import layers

_all_layers = [
    layers.StandardNormLayer,
    layers.StandardDenormLayer,
    layers.MaxFeatureStdNormLayer,
    layers.MaxFeatureStdNormLayer,
    layers.MeanFeatureStdNormLayer,
    layers.MeanFeatureStdDenormLayer,
]


def _get_tensor():
    """
    Tensor with 2 features (columns)
    and 2 samples (rows)
    """

    return tf.Variable([[0.0, 0.0], [1.0, 2.0]], dtype=tf.float32)


@pytest.mark.parametrize(
    "norm_cls, denorm_cls, expected",
    [
        (
            layers.StandardNormLayer,
            layers.StandardDenormLayer,
            [[-1.0, -1.0], [1.0, 1.0]],
        ),
        (
            layers.MaxFeatureStdNormLayer,
            layers.MaxFeatureStdDenormLayer,
            [[-0.5, -1.0], [0.5, 1.0]],
        ),
    ],
)
def test_normalize_layers(norm_cls, denorm_cls, expected):
    tensor = _get_tensor()
    norm_layer = norm_cls()
    denorm_layer = denorm_cls()
    norm_layer.fit(tensor)
    denorm_layer.fit(tensor)

    norm = norm_layer(tensor)
    expected = np.array(expected)
    np.testing.assert_allclose(norm, expected, rtol=1e-6)
    denorm = denorm_layer(norm)
    np.testing.assert_allclose(denorm, tensor, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    "norm_cls, denorm_cls",
    [
        (layers.StandardNormLayer, layers.StandardDenormLayer,),
        (layers.MeanFeatureStdNormLayer, layers.MeanFeatureStdDenormLayer,),
        (layers.MaxFeatureStdNormLayer, layers.MaxFeatureStdDenormLayer,),
    ],
)
@pytest.mark.parametrize("n", [3, 5])
def test_normalize_nd_layers(norm_cls, denorm_cls, n: int):
    array = np.random.randn(*[3 for _ in range(n - 1)], 1) * 2.0 + 5.0
    tensor = tf.Variable(array, dtype=tf.float32)
    norm_layer = norm_cls()
    denorm_layer = denorm_cls()
    norm_layer.fit(tensor)
    denorm_layer.fit(tensor)

    norm = norm_layer(tensor)
    np.testing.assert_almost_equal(np.mean(norm), 0.0, decimal=6)
    np.testing.assert_almost_equal(np.std(norm), 1.0, decimal=6)
    np.testing.assert_allclose(denorm_layer(norm_layer(tensor)), tensor, rtol=1e-5)


@pytest.mark.parametrize("layer_cls", _all_layers)
def test_layers_no_trainable_variables(layer_cls):
    tensor = _get_tensor()
    layer = layer_cls()
    layer(tensor)

    assert [] == layer.trainable_variables


def test_standard_layers_gradient_works_epsilon():
    tensor = _get_tensor()
    norm_layer = layers.StandardNormLayer()

    with tf.GradientTape(persistent=True) as tape:
        y = norm_layer(tensor)

    g = tape.gradient(y, tensor)
    expected = 1 / (norm_layer.sigma + norm_layer.epsilon)
    np.testing.assert_array_almost_equal(expected, g[0, :])


@pytest.mark.parametrize("layer_cls", _all_layers)
def test_fit_layers_are_fitted(layer_cls):
    tensor = _get_tensor()
    layer = layer_cls()

    assert not layer.fitted
    layer.fit(tensor)
    assert layer.fitted
