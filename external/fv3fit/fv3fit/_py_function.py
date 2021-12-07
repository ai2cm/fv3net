from typing import Any, Callable, Mapping

import tensorflow as tf


def py_function_dict_output(
    transform: Callable[..., Mapping[str, tf.Tensor]],
    inp: Any,
    signature: Mapping[str, tf.TensorSpec],
) -> Mapping[str, tf.Tensor]:
    """Wrap a user python function returning a dictionary of tensors for use in the
    tensorflow graph

    tf.data.Dataset.map only works with user functions that have been wrapped
    with tf.py_function, unfortunately, tf.py_function is not compatible with
    dictionary outputs or outputs with shapes.

    Args:
        transform: python function to embed in tensorflow graph
        inp: input tensors (see ``tf.py_function`` docs)
        signature: description of outputs of transform
    """

    keys = list(signature)

    def transform_values(*args):
        return tuple([transform(*args)[v] for v in keys])

    # py_function only works with tuple outputs, and no shapes
    out = tf.py_function(transform_values, inp, Tout=[signature[v].dtype for v in keys])
    return {
        v: tf.ensure_shape(out_i, signature[v].shape) for v, out_i in zip(keys, out)
    }
