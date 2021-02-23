import tensorflow.keras.layers as layers
import tensorflow as tf
import numpy as np
from ..._shared.scaler import StandardScaler


class StandardNormalize(layers.Layer):
    """x -> (x - mean) / std"""

    def __init__(self, *, mean, std):
        super().__init__()
        self.mean = tf.constant(mean)
        self.std = tf.constant(std)

    def call(self, inputs):
        return tf.transpose(
            layers.multiply(
                [layers.subtract([tf.transpose(inputs), self.mean]), 1.0 / self.std]
            )
        )

    def get_config(self):
        return {
            "mean": tf.make_ndarray(tf.make_tensor_proto(self.mean)),
            "std": tf.make_ndarray(tf.make_tensor_proto(self.std)),
        }


class StandardDenormalize(layers.Layer):
    """x -> x * std + mean"""

    def __init__(self, *, mean, std):
        super().__init__()
        self.mean = tf.constant(mean)
        self.std = tf.constant(std)

    def call(self, inputs):
        return tf.transpose(
            layers.add([layers.multiply([tf.transpose(inputs), self.std]), self.mean])
        )

    def get_config(self):
        return {
            "mean": tf.make_ndarray(tf.make_tensor_proto(self.mean)),
            "std": tf.make_ndarray(tf.make_tensor_proto(self.std)),
        }


class LayerStandardScaler(StandardScaler):
    def __init__(self):
        super().__init__()
        self._normalize_layer = None
        self._denormalize_layer = None

    @property
    def normalize_layer(self) -> layers.Layer:
        if self._normalize_layer is None:
            self._normalize_layer = StandardNormalize(mean=self.mean, std=self.std)
        return self._normalize_layer

    @property
    def scale_layer(self) -> layers.Layer:
        """Scale by standard deviation, but do not subtract the mean"""
        if self._normalize_layer is None:
            self._normalize_layer = StandardNormalize(
                mean=np.zeros_like(self.mean), std=self.std
            )
        return self._normalize_layer

    @property
    def denormalize_layer(self) -> layers.Layer:
        if self._denormalize_layer is None:
            self._denormalize_layer = StandardDenormalize(mean=self.mean, std=self.std)
        return self._denormalize_layer
