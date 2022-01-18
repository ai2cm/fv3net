import fv3fit
import numpy as np
import tensorflow as tf


def test_py_function_dict_output():
    def func(x):
        return {"a": tf.ones((1, 10)), "b": tf.ones((1, 10))}

    x = tf.ones([1, 1])
    expected = func(x)
    signature = {
        "a": tf.TensorSpec(shape=(1, 10), dtype=tf.float32),
        "b": tf.TensorSpec(shape=(1, 10), dtype=tf.float32),
    }

    def wrapped_func(x):
        return fv3fit.py_function_dict_output(func, [x], signature=signature)

    # check that it works...pretty trivial
    y = wrapped_func(x)
    assert set(y) == set(expected)
    for k in y:
        np.testing.assert_array_equal(y[k], expected[k])

    # check integration with tf.data...non-trivial (just calling ``func`` won't
    # always work)
    tf_ds = tf.data.Dataset.from_tensor_slices([1, 1]).map(wrapped_func)
    assert signature == tf_ds.element_spec
