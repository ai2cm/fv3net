from typing import Sequence
import fv3fit
import numpy as np
import tensorflow as tf
import pytest
import sys
from fv3fit.testing import numpy_print_precision


def print_result(result: fv3fit.ConvolutionalNetwork, decimals: int, file=sys.stdout):
    with numpy_print_precision(precision=decimals):
        print(f"output: {result.output.numpy()}", file=file)
        for i, hidden in enumerate(result.hidden_outputs):
            print(f"hidden {i}: {hidden.numpy()}", file=file)


def test_default_output(regtest):
    """default output should not change"""
    fv3fit.set_random_seed(0)
    config = fv3fit.ConvolutionalNetworkConfig()
    array = np.random.randn(2, 10, 10, 3)
    out = config.build(array, n_features_out=3)
    print_result(out, decimals=4, file=regtest)


def test_output_type():
    config = fv3fit.ConvolutionalNetworkConfig()
    array = np.random.randn(2, 10, 10, 3)
    convolutional_network = config.build(array, n_features_out=3)
    assert isinstance(convolutional_network, fv3fit.ConvolutionalNetwork)
    assert isinstance(convolutional_network.output, tf.Tensor)
    assert isinstance(convolutional_network.hidden_outputs, Sequence)
    assert all(
        isinstance(item, tf.Tensor) for item in convolutional_network.hidden_outputs
    )


@pytest.mark.parametrize(
    "input_shape, kernel_size, depth, features_out, base_output_shape",
    [
        pytest.param((3, 10, 10, 2), 3, 1, 5, (3, 10, 10), id="no_convolutions"),
        pytest.param((3, 10, 10, 2), 3, 2, 5, (3, 8, 8), id="one_convolution"),
        pytest.param(
            (3, 10, 10, 2), 5, 2, 5, (3, 6, 6), id="one_convolution_larger_kernel"
        ),
        pytest.param((3, 10, 10, 2), 3, 3, 5, (3, 6, 6), id="two_convolutions"),
        pytest.param(
            (3, 11, 15, 2),
            3,
            3,
            5,
            (3, 7, 11),
            id="two_convolutions_different_input_shape",
        ),
        pytest.param(
            (3, 10, 10, 2), 3, 2, 10, (3, 8, 8), id="one_convolution_more_filters"
        ),
    ],
)
def test_output_is_correct_shape(
    input_shape, kernel_size, depth, features_out, base_output_shape
):
    config = fv3fit.ConvolutionalNetworkConfig(kernel_size=kernel_size, depth=depth)
    array = np.random.randn(*input_shape)
    convolutional_network = config.build(array, n_features_out=features_out)
    output_shape = tuple(base_output_shape) + (features_out,)
    assert convolutional_network.output.shape == output_shape


@pytest.mark.parametrize("depth", [1, 2, 5])
@pytest.mark.parametrize("filters", [1, 5])
def test_hidden_layers_have_samples_equal_to_filters(depth, filters):
    input_shape = (3, 10, 10, 5)
    kernel_size = 3
    features_out = 1
    config = fv3fit.ConvolutionalNetworkConfig(
        kernel_size=kernel_size, depth=depth, filters=filters
    )
    array = np.random.randn(*input_shape)
    convolutional_network = config.build(array, n_features_out=features_out)
    for output in convolutional_network.hidden_outputs:
        assert output.shape[-1] == filters


@pytest.mark.parametrize(
    "kernel_size, depth, n_expected_changes",
    [
        pytest.param(3, 1, 1, id="no_convolutions"),
        pytest.param(3, 2, 9, id="one_convolution"),
        pytest.param(5, 2, 25, id="one_convolution_larger_filter"),
        pytest.param(3, 3, 25, id="two_convolutions"),
    ],
)
def test_modifying_one_input_modifies_correct_number_of_outputs(
    kernel_size, depth, n_expected_changes
):
    """
    Test that when you modify an input in the center of the domain,
    the correct number of output values change for the initialized network.
    """
    features_out = 1
    input_shape = (1, 17, 17, 1)
    config = fv3fit.ConvolutionalNetworkConfig(kernel_size=kernel_size, depth=depth)
    array = np.random.randn(*input_shape)
    input_layer = tf.keras.layers.Input(shape=input_shape[1:])
    convolutional_network = config.build(input_layer, n_features_out=features_out)
    model = tf.keras.Model(inputs=input_layer, outputs=convolutional_network.output)
    first_output = model.predict(array)
    array[0, 8, 8, 0] = np.random.randn()
    second_output = model.predict(array)
    assert np.sum(first_output != second_output) == n_expected_changes


def test_network_has_gaussian_noise_layer():
    fv3fit.set_random_seed(0)
    config = fv3fit.ConvolutionalNetworkConfig(gaussian_noise=0.1)
    n_features_in, n_features_out = 5, 5
    input = tf.keras.layers.Input(shape=(10, 10, n_features_in))
    convolutional_network = config.build(input, n_features_out=n_features_out)
    model = tf.keras.Model(inputs=input, outputs=convolutional_network.output)
    assert any(
        isinstance(layer, tf.keras.layers.GaussianNoise) for layer in model.layers
    )


@pytest.mark.parametrize("depth", [1, 2, 5])
def test_network_has_correct_number_of_hidden_layers(depth):
    fv3fit.set_random_seed(0)
    config = fv3fit.ConvolutionalNetworkConfig(depth=depth)
    n_features_in, n_features_out = 5, 5
    input = tf.keras.layers.Input(shape=(10, 10, n_features_in))
    convolutional_network = config.build(input, n_features_out=n_features_out)
    # one layer of depth is the output
    assert len(convolutional_network.hidden_outputs) == depth - 1
