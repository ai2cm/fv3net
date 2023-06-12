import numpy as np
import tensorflow as tf

from fv3fit.reservoir._reshaping import (
    flatten_2d_keeping_columns_contiguous,
    stack_array_preserving_last_dim,
    encode_columns,
    decode_columns,
    split_1d_into_2d_rows,
)


def test_flatten_2d_keeping_columns_contiguous():
    x = np.array([[1, 2], [3, 4], [5, 6]])
    np.testing.assert_array_equal(
        flatten_2d_keeping_columns_contiguous(x), np.array([1, 3, 5, 2, 4, 6])
    )


def test_stack_array_preserving_last_dim():
    z_profile = np.array([1, 2, 3, 4])
    x = np.array([[z_profile, z_profile, z_profile], [z_profile, z_profile, z_profile]])
    stacked_x = stack_array_preserving_last_dim(x)
    assert stacked_x.shape == (6, 4)
    np.testing.assert_array_equal(stacked_x[-1], z_profile)


def scalar_encoder(input_shapes, scalar):
    # input_X = tf.keras.layers.Input(shape=input_shape)
    inputs = [
        tf.keras.layers.Input(shape=input_shape, name=f"input_{i}")
        for i, input_shape in enumerate(input_shapes)
    ]
    full_input = tf.concat(inputs, axis=1)
    sc_mult = tf.keras.layers.Lambda(lambda x: x * scalar)(full_input)
    encoder = tf.keras.Model(inputs=inputs, outputs=sc_mult)
    return encoder


def var_data(nt, nx, ny, nz):
    z_profile = np.arange(nz)
    txyz = np.array(
        [
            [[z_profile * y * x * t for y in range(ny)] for x in range(nx)]
            for t in range(nt)
        ]
    )
    return txyz


def test_encode_columns():
    var_0_size, var_1_size = 2, 3
    nt, nx, ny = 8, 9, 10
    scalar = 2
    encoder = scalar_encoder([var_0_size, var_1_size], scalar=scalar)
    var0_data = var_data(nt, nx, ny, nz=var_0_size)
    var1_data = var_data(nt, nx, ny, nz=var_1_size)
    encoded = encode_columns([var0_data, var1_data], encoder)
    np.testing.assert_array_equal(encoded[:, :, :, :var_0_size], scalar * var0_data)
    np.testing.assert_array_equal(encoded[:, :, :, var_0_size:], scalar * var1_data)


def scalar_decoder(input_shape, scalar, n_outputs):
    # Returns list of n_outputs arrays, each is the input array * scalar
    input = tf.keras.layers.Input(shape=input_shape)
    sc_mult = [
        tf.keras.layers.Lambda(lambda x: x * scalar)(input) for n in range(n_outputs)
    ]
    decoder = tf.keras.Model(inputs=input, outputs=sc_mult)
    return decoder


def test_decode_columns():
    latent_var_size = 3
    nt, nx, ny = 8, 9, 10
    scalar = 2
    decoder = scalar_decoder(input_shape=latent_var_size, scalar=scalar, n_outputs=2)
    var0_data = var_data(nt, nx, ny, nz=latent_var_size)
    decoded_output = decode_columns(var0_data, decoder)
    np.testing.assert_array_equal(decoded_output[0], var0_data * scalar)
    np.testing.assert_array_equal(decoded_output[1], var0_data * scalar)


def test_split_1d_into_2d_rows():
    x = np.arange(12)
    x_2d = split_1d_into_2d_rows(x, n_rows=4)
    np.testing.assert_array_equal(
        x_2d, np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
    )
