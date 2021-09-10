from typing import Sequence
import fv3fit
import numpy as np
import tensorflow as tf
import pytest


def test_default_output(regtest):
    """default output should not change"""
    fv3fit.set_random_seed(0)
    config = fv3fit.DenseNetworkConfig()
    array = np.random.randn(5, 10)
    out = config.build(array, n_features_out=3)
    print(out, file=regtest)


def test_output_type():
    config = fv3fit.DenseNetworkConfig()
    array = np.random.randn(5, 10)
    dense_network = config.build(array, n_features_out=3)
    assert isinstance(dense_network, fv3fit.DenseNetwork)
    assert isinstance(dense_network.output, tf.Tensor)
    assert isinstance(dense_network.hidden_outputs, Sequence)
    assert all(isinstance(item, tf.Tensor) for item in dense_network.hidden_outputs)


@pytest.mark.parametrize(
    "n_samples, n_features, n_features_out",
    [
        pytest.param(5, 5, 5, id="all_same"),
        pytest.param(10, 5, 5, id="features_equals_features_out"),
        pytest.param(10, 5, 1, id="one_feature_out"),
        pytest.param(10, 5, 7, id="more_features_out"),
    ],
)
def test_output_is_correct_shape(n_samples, n_features, n_features_out):
    config = fv3fit.DenseNetworkConfig()
    array = np.random.randn(n_samples, n_features)
    dense_network = config.build(array, n_features_out=n_features_out)
    assert dense_network.output.shape == (n_samples, n_features_out)


def test_network_has_gaussian_noise_layer():
    fv3fit.set_random_seed(0)
    config = fv3fit.DenseNetworkConfig(gaussian_noise=0.1)
    n_features_in, n_features_out = 5, 5
    input = tf.keras.layers.Input(shape=(n_features_in,))
    dense_network = config.build(input, n_features_out=n_features_out)
    model = tf.keras.Model(inputs=input, outputs=dense_network.output)
    assert any(
        isinstance(layer, tf.keras.layers.GaussianNoise) for layer in model.layers
    )


@pytest.mark.parametrize("depth", [1, 2, 5])
def test_network_has_correct_number_of_hidden_layers(depth):
    fv3fit.set_random_seed(0)
    config = fv3fit.DenseNetworkConfig(depth=depth)
    n_features_in, n_features_out = 5, 5
    input = tf.keras.layers.Input(shape=(n_features_in,))
    dense_network = config.build(input, n_features_out=n_features_out)
    # one layer of depth is the output
    assert len(dense_network.hidden_outputs) == depth - 1
