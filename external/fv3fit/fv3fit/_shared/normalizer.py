from typing import BinaryIO
import tensorflow.keras.layers as layers
import tensorflow as tf
import numpy as np


class Scale(layers.Layer):
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


class InverseScale(layers.Layer):
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


class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None
        self._transform_layer = None
        self._inverse_transform_layer = None

    def fit(self, X):
        self.mean = X.mean(axis=0).astype(np.float32)
        self.std = X.std(axis=0).astype(np.float32)

    def transform(self, X):
        return (X - self.mean) / self.std

    def inverse_transform(self, X):
        return X * self.std + self.mean

    @property
    def transform_layer(self) -> layers.Layer:
        if self._transform_layer is None:
            self._transform_layer = Scale(mean=self.mean, std=self.std)
        return self._transform_layer

    @property
    def inverse_transform_layer(self) -> layers.Layer:
        if self._inverse_transform_layer is None:
            self._inverse_transform_layer = InverseScale(mean=self.mean, std=self.std)
        return self._inverse_transform_layer

    def dump(self, f: BinaryIO):
        return np.savez(f, mean=self.mean, std=self.std)

    @classmethod
    def load(cls, f: BinaryIO):
        data = np.load(f)
        scaler = cls()
        scaler.mean = data["mean"]
        scaler.std = data["std"]
        return scaler
