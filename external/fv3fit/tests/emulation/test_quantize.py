from fv3fit.emulation.transforms.quantize import (
    quantize_step,
    unquantize_step,
    StepQuantizerFactory,
)

import numpy as np
import tensorflow as tf


def test_quantize_step():
    bins = [-0.1, 1.0, 2.0]
    x = tf.convert_to_tensor([0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    d = tf.convert_to_tensor([0, 0.0, 1.0, -0.5, 1.5, 2.0, 2.5])
    expected_bin = [0, 1, 3, 2, 4, 4, 5]
    y = x + d
    bin_id = quantize_step(x, y, bins=bins)

    assert bin_id.numpy().tolist() == expected_bin


def test_unquantize_step():
    midpoints = [1.0, 2, 3, 4]
    x = tf.convert_to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    bin_id = tf.convert_to_tensor([0, 1, 3, 2, 4, 4, 5])
    d = [0, 0, midpoints[1], midpoints[0], midpoints[2], midpoints[2], midpoints[3]]
    d = tf.convert_to_tensor(d)

    out = unquantize_step(x, bin_id, midpoints)
    expected = x + d
    expected = expected.numpy()
    expected[0] = 0.0
    np.testing.assert_array_equal(out, expected)


def test_StepQuantizerFactory():
    tf.random.set_seed(0)
    n = 100
    x = tf.random.uniform([10])
    d = tf.random.uniform([10])
    y = tf.where(x < 0.5, 0, tf.where(x < 0.75, x, x + d))

    data = {"x": x, "y": y}
    factory = StepQuantizerFactory("x", "y", "b", n, 0, 1)
    transform = factory.build(data)

    f = transform.forward(data)
    assert f["b"].dtype == tf.int32
    b = transform.backward(f)

    np.testing.assert_allclose(y, b["y"], atol=1 / n / 2)
