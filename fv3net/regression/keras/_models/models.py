from typing import Sequence, Tuple, Iterable
import xarray as xr
import logging
import abc
import tensorflow as tf
from .._packer import ArrayPacker
import numpy as np
import os
from ._filesystem import get_dir, put_dir

logger = logging.getLogger(__file__)


class _XyArraySequence(tf.keras.utils.Sequence):
    """
    Wrapper object converting a sequence of batch datasets
    to a sequence of input/output numpy arrays.
    """

    def __init__(
        self,
        X_packer: ArrayPacker,
        y_packer: ArrayPacker,
        dataset_sequence: Sequence[xr.Dataset],
    ):
        self.X_packer = X_packer
        self.y_packer = y_packer
        self.dataset_sequence = dataset_sequence

    def __len__(self) -> int:
        return len(self.dataset_sequence)

    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray]:
        ds = self.dataset_sequence[idx]
        X = self.X_packer.to_array(ds)
        y = self.y_packer.to_array(ds)
        return X, y


class Model(abc.ABC):
    """
    Abstract base class for a machine learning model which operates on xarray
    datasets, and is trained on sequences of such datasets.
    """

    @abc.abstractmethod
    def __init__(
        self,
        sample_dim_name: str,
        input_variables: Iterable[str],
        output_variables: Iterable[str],
        **hyperparameters,
    ):
        super().__init__()

    @abc.abstractmethod
    def fit(self, X: Sequence[xr.Dataset]) -> None:
        pass

    @abc.abstractmethod
    def predict(self, X: xr.Dataset) -> xr.Dataset:
        pass

    @abc.abstractmethod
    def dump(self, path: str) -> None:
        """Serialize the model to a directory."""
        pass

    @abc.abstractmethod
    def load(self, path: str) -> object:
        """Load a serialized model from a directory."""
        pass


class PackedKerasModel(Model):
    """
    Abstract base class for a keras-based model which operates on xarray
    datasets containing a "sample" dimension (as defined by loaders.SAMPLE_DIM_NAME),
    where each variable has at most one non-sample dimension.

    Subclasses are defined primarily using a `get_model` method, which returns a
    Keras model.
    """

    # these should only be used in the dump/load routines for this class
    _MODEL_FILENAME = "model.tf"
    _X_PACKER_FILENAME = "X_packer.json"
    _Y_PACKER_FILENAME = "y_packer.json"

    def __init__(
        self,
        sample_dim_name: str,
        input_variables: Iterable[str],
        output_variables: Iterable[str],
    ):
        super().__init__(sample_dim_name, input_variables, output_variables)
        self._model = None
        self.X_packer = ArrayPacker(
            sample_dim_name=sample_dim_name, names=input_variables
        )
        self.y_packer = ArrayPacker(
            sample_dim_name=sample_dim_name, names=output_variables
        )

    @property
    def model(self) -> tf.keras.Model:
        if self._model is None:
            raise RuntimeError("must call fit() for keras model to be available")
        return self._model

    @abc.abstractmethod
    def get_model(self, features_in: int, features_out: int) -> tf.keras.Model:
        """Returns a Keras model to use as the underlying predictive model.

        Args:
            features_in: the number of input features
            features_out: the number of output features

        Returns:
            model: a Keras model whose input shape is [n_samples, features_in] and
                output shape is [n_samples, features_out]
        """
        pass

    def fit(self, batches: Sequence[xr.Dataset]) -> None:
        Xy = _XyArraySequence(self.X_packer, self.y_packer, batches)
        if self._model is None:
            features_in, features_out = count_batch_features(Xy)
            self._model = self.get_model(features_in, features_out)
        self.fit_array(Xy)

    def fit_array(self, X: Sequence[Tuple[np.ndarray, np.ndarray]]) -> None:
        return self.model.fit(X)

    def predict(self, X: xr.Dataset) -> xr.Dataset:
        return self.y_packer.to_dataset(self.predict_array(self.X_packer.to_array(X)))

    def predict_array(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def dump(self, path: str) -> None:
        with put_dir(path) as path:
            model_filename = os.path.join(path, self._MODEL_FILENAME)
            self.model.save(model_filename)
            with open(os.path.join(path, self._X_PACKER_FILENAME), "w") as f:
                self.X_packer.dump(f)
            with open(os.path.join(path, self._Y_PACKER_FILENAME), "w") as f:
                self.y_packer.dump(f)

    @classmethod
    def load(cls, path: str) -> Model:
        with get_dir(path) as path:
            print(os.listdir(path))
            with open(os.path.join(path, cls._X_PACKER_FILENAME), "r") as f:
                X_packer = ArrayPacker.load(f)
            with open(os.path.join(path, cls._Y_PACKER_FILENAME), "r") as f:
                y_packer = ArrayPacker.load(f)
            obj = cls(X_packer.sample_dim_name, X_packer.names, y_packer.names)
            model_filename = os.path.join(path, cls._MODEL_FILENAME)
            obj._model = tf.keras.models.load_model(model_filename)
            obj.X_packer = X_packer
            obj.y_packer = y_packer
            return obj


def count_batch_features(
    batches: Sequence[Tuple[np.ndarray, np.ndarray]]
) -> Tuple[int, int]:
    """Returns the number of input and output features in the first batch of a sequence.
    """
    features_in = batches[0][0].shape[-1]
    features_out = batches[0][1].shape[-1]
    return features_in, features_out


class DenseModel(PackedKerasModel):
    """
    A simple feedforward neural network model with dense layers.
    """

    def __init__(
        self,
        sample_dim_name: str,
        input_variables: Iterable[str],
        output_variables: Iterable[str],
        depth: int = 3,
        width: int = 16,
    ):
        self.width = width
        self.depth = depth
        super().__init__(sample_dim_name, input_variables, output_variables)

    def get_model(self, features_in: int, features_out: int) -> tf.keras.Model:
        model = tf.keras.Sequential()
        model.add(tf.keras.Input(features_in))
        for i in range(self.depth - 1):
            model.add(
                tf.keras.layers.Dense(self.width, activation=tf.keras.activations.relu)
            )
        model.add(
            tf.keras.layers.Dense(features_out, activation=tf.keras.activations.relu)
        )
        model.compile(optimizer="sgd", loss="mse")
        return model
