from typing import BinaryIO
import tensorflow.keras.layers as layers
import numpy as np


class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = X.mean(axis=0).astype(np.float32)
        self.std = X.std(axis=0).astype(np.float32)

    def transform(self, X):
        return (X - self.mean) / self.std

    def inverse_transform(self, X):
        return X * self.std + self.mean

    def transform_layer(self, layer: layers.Layer) -> layers.Layer:
        return layers.multiply(
            [layers.subtract([layer, self.mean[None, :]]), 1.0 / self.std[None, :]]
        )

    def inverse_transform_layer(self, layer: layers.Layer) -> layers.Layer:
        return layers.add(
            [layers.multiply([layer, self.std[None, :]]), self.mean[None, :]]
        )

    def dump(self, f: BinaryIO):
        return np.savez(f, mean=self.mean, std=self.std)

    @classmethod
    def load(cls, f: BinaryIO):
        data = np.load(f)
        scaler = cls()
        scaler.mean = data["mean"]
        scaler.std = data["std"]
        return scaler
