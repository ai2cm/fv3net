from typing import BinaryIO
import tensorflow.keras.layers as layers
import tensorflow as tf
import numpy as np


class Normalize(layers.Layer):
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


class Denormalize(layers.Layer):
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
        self._normalize_layer = None
        self._denormalize_layer = None

    def fit(self, X):
        self.mean = X.mean(axis=0).astype(np.float32)
        self.std = X.std(axis=0).astype(np.float32)

    def normalize(self, X):
        return (X - self.mean) / self.std

    def denormalize(self, X):
        return X * self.std + self.mean

    @property
    def normalize_layer(self) -> layers.Layer:
        if self._normalize_layer is None:
            self._normalize_layer = Normalize(mean=self.mean, std=self.std)
        return self._normalize_layer

    @property
    def denormalize_layer(self) -> layers.Layer:
        if self._denormalize_layer is None:
            self._denormalize_layer = Denormalize(mean=self.mean, std=self.std)
        return self._denormalize_layer

    def dump(self, f: BinaryIO):
        data = {}
        if self.mean is not None:
            data["mean"] = self.mean
        if self.std is not None:
            data["std"] = self.std
        return np.savez(f, **data)

    @classmethod
    def load(cls, f: BinaryIO):
        data = np.load(f)
        scaler = cls()
        scaler.mean = data.get("mean")
        scaler.std = data.get("std")
        return scaler
