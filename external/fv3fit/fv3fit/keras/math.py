import tensorflow as tf


@tf.function
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
    index = tf.searchsorted(x, xg, side="right") - 1
    index = tf.maximum(index, 0)
    return tf.gather(y, index)
