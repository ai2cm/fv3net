from typing import Sequence, Iterable
import xarray as xr
import logging
import numpy as np
import abc
import joblib
import loaders
import tensorflow as tf


def to_array_sequence(dataset_sequence: Sequence[xr.Dataset], varnames: Iterable[str]) -> Sequence[np.ndarray]:
    def dataset_to_array(dataset):
        return _pack(dataset[varnames], loaders.SAMPLE_DIM_NAME)
    return loaders.FunctionOutputSequence(dataset_sequence, dataset_to_array)


logger = logging.getLogger(__file__)


class ArrayPacker:

    def __init__(self):
        pass

    def pack(self, data: xr.Dataset) -> np.ndarray:
        pass

    def unpack(self, array: np.ndarray) -> xr.Dataset:
        pass


class Model(abc.ABC):

    @abc.abstractmethod
    def fit(self, X: Sequence[xr.Dataset], y: Sequence[xr.Dataset]):
        pass

    @abc.abstractmethod
    def predict(self, X: xr.Dataset) -> xr.Dataset:
        pass

    def dump(self, file):
        joblib.dump(self, file)

    def load(self, file):
        return joblib.load(file)


class DenseModel(Model):

    def __init__(self, depth=3, width=16, **hyperparameters):
        self.width = width
        self.depth = depth
        self._model = None

    @property
    def model(self):
        if self._model is None:
            model = tf.keras.Sequential()
            for i in range(self.depth):
                model.add(
                    tf.keras.layers.Dense(self.width, activation=tf.keras.activations.relu)
                )
            model.compile(optimizer="sgd", loss="mse")
            self._model = model
        return self._model

    def fit(self, X, y):
        self.model.fit(X, y)
