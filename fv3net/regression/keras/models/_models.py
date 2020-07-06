from typing import Sequence, Tuple
import xarray as xr
import logging
import abc
import tensorflow as tf
from ..packer import ArrayPacker
import numpy as np
import os

logger = logging.getLogger(__file__)


class _SampleSequence(tf.keras.utils.Sequence):
    def __init__(self, X_packer, y_packer, dataset_sequence):
        self.X_packer = X_packer
        self.y_packer = y_packer
        self.dataset_sequence = dataset_sequence

    def __len__(self):
        return len(self.dataset_sequence)

    def __getitem__(self, idx):
        ds = self.dataset_sequence[idx]
        X = self.X_packer.pack(ds)
        y = self.y_packer.pack(ds)
        return X, y


def sample_from_dataset(packer, dataset):
    return packer.pack_X(dataset), packer.pack_y(dataset)


class Model(abc.ABC):
    @abc.abstractmethod
    def fit(self, X: Sequence[xr.Dataset], y: Sequence[xr.Dataset]):
        pass

    @abc.abstractmethod
    def predict(self, X: xr.Dataset) -> xr.Dataset:
        pass

    @abc.abstractmethod
    def dump(self, file):
        pass

    @abc.abstractmethod
    def load(self, file):
        pass


class PackedKerasModel(Model):

    MODEL_FILENAME = "model.tf"
    X_PACKER_FILENAME = "X_packer.json"
    Y_PACKER_FILENAME = "y_packer.json"

    def __init__(self, input_variables, output_variables):
        super().__init__()
        self._model = None
        self.X_packer = ArrayPacker(input_variables)
        self.y_packer = ArrayPacker(output_variables)

    @property
    def model(self) -> tf.keras.Model:
        if self._model is None:
            raise RuntimeError("must call fit() for keras model to be available")
        return self._model

    @abc.abstractmethod
    def get_model(self, features_in: int, features_out: int) -> tf.keras.Model:
        pass

    def fit(
        self, batches: Sequence[xr.Dataset],
    ):
        X = _SampleSequence(self.X_packer, self.y_packer, batches)
        if self._model is None:
            features_in = X[0][0].shape[-1]
            features_out = X[0][1].shape[-1]
            self._model = self.get_model(features_in, features_out)
        self.fit_array(X)

    def fit_array(self, X: Sequence[Tuple[np.ndarray, np.ndarray]]):
        return self.model.fit(X)

    def predict(self, X: xr.Dataset) -> xr.Dataset:
        return self.y_packer.unpack(self.predict_array(self.X_packer.pack(X)))

    def predict_array(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def dump(self, path):
        if os.path.isfile(path):
            raise ValueError(f"path {path} exists and is not a directory")
        model_filename = os.path.join(path, self.MODEL_FILENAME)
        self.model.save(model_filename)
        with open(os.path.join(path, self.X_PACKER_FILENAME), "w") as f:
            self.X_packer.dump(f)
        with open(os.path.join(path, self.Y_PACKER_FILENAME), "w") as f:
            self.y_packer.dump(f)

    @classmethod
    def load(cls, path):
        with open(os.path.join(path, cls.X_PACKER_FILENAME), "r") as f:
            X_packer = ArrayPacker.load(f)
        with open(os.path.join(path, cls.Y_PACKER_FILENAME), "r") as f:
            y_packer = ArrayPacker.load(f)
        obj = cls(X_packer.names, y_packer.names)
        model_filename = os.path.join(path, cls.MODEL_FILENAME)
        obj._model = tf.keras.models.load_model(model_filename)
        obj.X_packer = X_packer
        obj.y_packer = y_packer
        return obj


class DenseModel(PackedKerasModel):
    def __init__(
        self, input_variables, output_variables, depth=3, width=16, **hyperparameters
    ):
        self.width = width
        self.depth = depth
        super().__init__(input_variables, output_variables)

    def get_model(self, features_in: int, features_out: int) -> tf.keras.Model:
        model = tf.keras.Sequential()
        model.add(tf.keras.Input(features_in))
        model.add(tf.keras.layers.BatchNormalization())
        for i in range(self.depth - 1):
            model.add(
                tf.keras.layers.Dense(self.width, activation=tf.keras.activations.relu)
            )
        model.add(
            tf.keras.layers.Dense(features_out, activation=tf.keras.activations.relu)
        )
        model.compile(optimizer="sgd", loss="mse")
        return model
