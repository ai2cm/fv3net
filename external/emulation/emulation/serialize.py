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


def get_parser(data: Mapping[str, Tensor]):
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
