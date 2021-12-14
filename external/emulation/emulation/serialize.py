from typing import Mapping, Union
import tensorflow as tf
import numpy as np

Tensor = Union[np.ndarray, tf.Tensor]


def _bytes_feature(b: bytes):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[b]))


def _tensor_feature(tensor: Tensor):
    return _bytes_feature(tf.io.serialize_tensor(tensor).numpy())


def serialize_tensor_dict(data: Mapping[str, Tensor]) -> bytes:
    example = tf.train.Example(
        features=tf.train.Features(
            feature={key: _tensor_feature(val) for key, val in data.items()}
        )
    )

    return example.SerializeToString()


def get_parser(data: Mapping[str, Tensor]) -> tf.Module:
    """Returns a ``tensorflow`` module object for parsing byte records saved by
    ``serialize_tensor_dict``.


    Args:
        data: a dataset object to use as a template

    Returns:
        a parser object with the following methods:

        - ``parse_single_example``: parser a single bytes record into a
                dictionary of tensors. Analogous to ``tf.io.parse_single_example``_.
        - ``parse_example``: goes from a vector of bytes records to a dictionary
                of tensors stacked along their first dimension. The length of
                this dimension is the length of the input vector. Analogous to
                ``tf.io.parse_example``_. Often more performance than
                ``parse_single_example`` for small record sizes.

        This module object can be serialized and loaded by
        ``tf.saved_model.{save/load}`` in any environment with tensorflow
        installed.

    Examples:

        >>> data = {"a": tf.ones((4,)), "b": tf.constant(1)}
        >>> serialized = serialize_tensor_dict(data)
        >>> parser = get_parser(data)
        >>> single_record = tf.constant(serialized)
        >>> single_record.shape, single_record.dtype
        (TensorShape([]), tf.string)
        >>> example = parser.parse_single_example(single_record)
        >>> example["a"]
        <tf.Tensor: shape=(4,), dtype=float32, numpy=array([1., 1., 1., 1.], \
dtype=float32)>
        >>> example["b"]
        <tf.Tensor: shape=(), dtype=int32, numpy=1>
        >>> records = tf.repeat(tf.constant(serialized), (10,))
        >>> records.shape, records.dtype
        (TensorShape([10]), tf.string)
        >>> vectorized_parse = parser.parse_example(records)
        >>> vectorized_parse["a"].shape
        TensorShape([10, 4])

    .. _``tf.io.parse_single_example``:: https://www.tensorflow.org/api_docs/python/tf/io/parse_single_example # noqa
    .. _``tf.io.parse_example``:: https://www.tensorflow.org/api_docs/python/tf/io/parse_example # noqa

    """
    features = {
        key: tf.io.FixedLenFeature([], tf.string, default_value=b"") for key in data
    }
    dtypes = {key: val.dtype for key, val in data.items()}

    def _get_shape(shape):
        if shape == ():
            return ()
        else:
            return [None] + shape[1:]

    sizes = {key: _get_shape(tensor.shape) for key, tensor in data.items()}

    class Parser(tf.Module):
        @tf.function
        def _parse_dict_of_bytes(self, x: Mapping[str, tf.Tensor]):
            return {
                key: tf.ensure_shape(
                    tf.io.parse_tensor(x[key], dtypes[key]), sizes[key]
                )
                for key in sizes
            }

        @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string)])
        def parse_example(self, records: tf.Tensor) -> tf.data.Dataset:
            parsed = tf.io.parse_example(records, features)
            return tf.map_fn(
                self._parse_dict_of_bytes,
                parsed,
                fn_output_signature={
                    key: tf.TensorSpec(sizes[key], dtypes[key]) for key in sizes
                },
            )
            ds = tf.data.Dataset.from_tensor_slices(parsed)
            return ds.map(self._parse_dict_of_bytes)

        @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
        def parse_single_example(self, record: tf.Tensor):
            parsed = tf.io.parse_single_example(record, features)
            return self._parse_dict_of_bytes(parsed)

    return Parser()


if __name__ == "__main__":
    # Got error when trying to invoke the doctests from pytest::
    # $ pytest --doctest-modules
    # INTERNALERROR>     if "regtest" not in item.fixturenames:
    # INTERNALERROR> AttributeError: 'DoctestItem' object has no attribute 'fixturenames' # noqa

    import doctest  # noqa

    doctest.testmod()
