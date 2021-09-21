import numpy as np
import tensorflow as tf

from fv3fit.emulation.layers import IncrementStateLayer, PassThruLayer


def test_increment_layer():

    in_ = tf.ones((2, 4), dtype=tf.float32)
    incr = tf.ones((2, 4), dtype=tf.float32)
    expected = tf.convert_to_tensor([[3] * 4, [3] * 4], dtype=tf.float32)

    incr_layer = IncrementStateLayer(2)
    incremented = incr_layer([in_, incr])

    assert incr_layer.dt_sec == 2
    np.testing.assert_array_equal(incremented, expected)


def test_passthru_layer():

    in_ = tf.ones((2, 4), dtype=tf.float32)
    expected = tf.convert_to_tensor([[1] * 4] * 2, dtype=tf.float32)

    pass_layer = PassThruLayer()
    result = pass_layer(in_)

    np.testing.assert_array_equal(result, expected)
