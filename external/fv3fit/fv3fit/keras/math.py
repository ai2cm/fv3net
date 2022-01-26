from typing import Callable
import tensorflow as tf


def piecewise(x: tf.Tensor, y: tf.Tensor, xg: tf.Tensor) -> tf.Tensor:
    """1D 0th order interpolation

    Args:
        x: original ordinate, 1D, must be sorted
        y:  original value, 1D
        xg: points to interpolate at

    Return:
        yg: interpolated values. same shape as xg. Uses constant extrapolation.
            Within the min/max of x it has the following form::

                f(xg) = y[i] if x[i] <= xg < x[i+1]
    """
    xg_flat = tf.reshape(xg, [-1])
    index = tf.searchsorted(x, xg_flat, side="right") - 1
    index = tf.maximum(index, 0)
    yg = tf.gather(y, index)
    return tf.reshape(yg, tf.shape(xg))


def groupby_bins(
    edges: tf.Tensor,
    x: tf.Tensor,
    y: tf.Tensor,
    reduction: Callable[[tf.Tensor], tf.Tensor],
) -> tf.Tensor:
    """Groupby edges (left inclusive)"""

    assert tf.rank(edges).numpy() == 1, edges
    assert tf.reduce_all(edges[1:] - edges[:-1] > 0), "edges not sorted"
    assert x.shape == y.shape

    n = edges.shape[0] - 1

    output = [reduction(y[(edges[i] <= x) & (x < edges[i + 1])]) for i in range(n)]

    return tf.stack(output)
