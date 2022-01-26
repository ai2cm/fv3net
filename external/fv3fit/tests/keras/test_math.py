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
    out = groupby_bins(edges, values, values, tf.size)
    assert [2, 3] == out.numpy().tolist()
