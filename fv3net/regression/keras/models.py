from typing import Union, Sequence, Iterable, Tuple
import collections
import xarray as xr
import logging
import numpy as np
import abc
import joblib
import loaders
import tensorflow as tf
from ..sklearn.wrapper import _pack, _unpack

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
        print(idx, X.shape, y.shape)
        return X, y


def pack(dataset) -> Tuple[np.ndarray, np.ndarray]:
    return _pack(dataset, loaders.SAMPLE_DIM_NAME)


def unpack(dataset, feature_indices):
    return _unpack(dataset, loaders.SAMPLE_DIM_NAME, feature_indices)


def sample_from_dataset(packer, dataset):
    return packer.pack_X(dataset), packer.pack_y(dataset)


class ArrayPacker:
    def __init__(self, names):
        self._indices = None
        self._names = names

    @property
    def names(self):
        return self._names

    def pack(self, dataset: xr.Dataset) -> np.ndarray:
        packed, indices = pack(dataset[self.names])
        if self._indices is None:
            self._indices = indices
        return packed

    def unpack(self, array: np.ndarray) -> xr.Dataset:
        if self._indices is None:
            raise RuntimeError(
                "must pack at least once before unpacking, "
                "so dimension lengths are known"
            )
        return unpack(array, self._indices)


class Model(abc.ABC):
    def __init__(self, input_variables, output_variables, *args, **kwargs):
        self.X_packer = ArrayPacker(input_variables)
        self.y_packer = ArrayPacker(output_variables)

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


class DenseModel(Model, ArrayPacker):
    def __init__(
        self, input_variables, output_variables, depth=3, width=16, **hyperparameters
    ):
        self.width = width
        self.depth = depth
        self._model = None
        super().__init__(input_variables, output_variables)

    @property
    def model(self):
        return self._model

    def _init_model(self, features_in: int, features_out: int) -> tf.keras.Model:
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
        self._model = model

    def fit(
        self, batches: Sequence[xr.Dataset],
    ):
        X = _SampleSequence(self.X_packer, self.y_packer, batches)
        features_in = X[0][0].shape[-1]
        features_out = X[0][1].shape[-1]
        self._init_model(features_in, features_out)
        self.model.fit(X)

    def predict(self, X: xr.Dataset) -> xr.Dataset:
        return self.y_packer.unpack(self.model.predict(self.X_packer.pack(X)))
