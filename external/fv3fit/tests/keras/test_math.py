import tensorflow as tf
import numpy as np

from fv3fit.keras.math import piecewise, groupby_bins


def test_piecewise_interp():

    x = tf.convert_to_tensor([0.0, 1, 2])
    y = 2.0 * x
    xg = tf.convert_to_tensor([-2, 0.5, 0.75, 1, 1.5, 2.5])

    yg = piecewise(x, y, xg)
    expected = tf.gather(y, [0, 0, 0, 1, 1, 2])
    np.testing.assert_array_almost_equal(expected, yg.numpy())


def test_piecewise_multiple_dimensions():
    x = tf.convert_to_tensor([0.0, 1, 2])
    y = 2.0 * x
    xg = tf.ones((10, 10))
    yg = piecewise(x, y, xg)
    assert xg.shape == yg.shape


def test_groupby_bins():
    edges = tf.convert_to_tensor([0.0, 1, 4])
    values = tf.convert_to_tensor([0.5, 0.75, 1, 2, 3])
    out = groupby_bins(edges, values, values, lambda x, w: tf.size(x[w]))
    assert [2, 3] == out.numpy().tolist()


def test_groupby_bins_y_has_more_dims_than_x():
    """groupby should work if y and x are broadcastable."""
    num_channels = 4

    edges = tf.convert_to_tensor([0.0, 1, 4])
    x = tf.convert_to_tensor([0.5, 0.75, 1, 2, 3])
    y = tf.convert_to_tensor([0.5, 0.75, 1, 2, 3])
    y = tf.tile(y[:, None], tf.convert_to_tensor([1, num_channels]))

    def reduce(x, w):
        return tf.reduce_mean(x, axis=[0])

    out = groupby_bins(edges, x, y, reduction=reduce)
    assert tuple(out.shape) == (2, num_channels)
