from typing import Sequence, Tuple, Iterable
import xarray as xr
import logging
import abc
import tensorflow as tf
from ..._shared import ArrayPacker
import numpy as np
import os
from ._filesystem import get_dir, put_dir
from ..._shared import StandardScaler

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
    def fit(self, batches: Sequence[xr.Dataset]) -> None:
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
    _X_SCALER_FILENAME = "X_scaler.npz"
    _Y_SCALER_FILENAME = "y_scaler.npz"

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
        self.X_scaler = StandardScaler()
        self.y_scaler = StandardScaler()

    @property
    def model(self) -> tf.keras.Model:
        if self._model is None:
            raise RuntimeError("must call fit() for keras model to be available")
        return self._model

    def _fit_normalization(self, X: np.ndarray, y: np.ndarray):
        self.X_scaler.fit(X)
        self.y_scaler.fit(y)

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
            X, y = Xy[0]
            features_in, features_out = X.shape[-1], y.shape[-1]
            self._fit_normalization(X, y)
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
            with open(os.path.join(path, self._X_SCALER_FILENAME), "wb") as f_binary:
                self.X_scaler.dump(f_binary)
            with open(os.path.join(path, self._Y_SCALER_FILENAME), "wb") as f_binary:
                self.y_scaler.dump(f_binary)

    @classmethod
    def load(cls, path: str) -> Model:
        with get_dir(path) as path:
            with open(os.path.join(path, cls._X_PACKER_FILENAME), "r") as f:
                X_packer = ArrayPacker.load(f)
            with open(os.path.join(path, cls._Y_PACKER_FILENAME), "r") as f:
                y_packer = ArrayPacker.load(f)
            with open(os.path.join(path, cls._X_SCALER_FILENAME), "rb") as f_binary:
                X_scaler = StandardScaler.load(f_binary)
            with open(os.path.join(path, cls._Y_SCALER_FILENAME), "rb") as f_binary:
                y_scaler = StandardScaler.load(f_binary)
            obj = cls(X_packer.sample_dim_name, X_packer.names, y_packer.names)
            model_filename = os.path.join(path, cls._MODEL_FILENAME)
            obj._model = tf.keras.models.load_model(model_filename)
            obj.X_packer = X_packer
            obj.y_packer = y_packer
            obj.X_scaler = X_scaler
            obj.y_scaler = y_scaler
            return obj


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
        inputs = tf.keras.Input(features_in)
        x = self.X_scaler.transform_layer(inputs)
        for i in range(self.depth - 1):
            x = tf.keras.layers.Dense(self.width, activation=tf.keras.activations.relu)(
                x
            )
        x = tf.keras.layers.Dense(features_out, activation=tf.keras.activations.relu)(x)
        outputs = self.y_scaler.inverse_transform_layer(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer="sgd", loss="mse")
        return model
