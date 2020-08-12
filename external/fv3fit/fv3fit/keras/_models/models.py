from typing import Sequence, Tuple, Iterable, Mapping, Union, Optional
import xarray as xr
import logging
import abc
import tensorflow as tf
from ..._shared import ArrayPacker
import numpy as np
import os
from ._filesystem import get_dir, put_dir
from .normalizer import StandardScaler
from .loss import get_weighted_mse
import yaml

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

    def __init__(
        self,
        sample_dim_name: str,
        input_variables: Iterable[str],
        output_variables: Iterable[str],
        **hyperparameters,
    ):
        """Initialize the model.

        Args:
            sample_dim_name: name of the sample dimension in datasets used as
                inputs and outputs.
            input_variables: names of input variables
            output_variables: names of output variables
            hyperparameters: options for subclasses
        """
        if len(hyperparameters) > 0:
            raise TypeError(
                "Model base class received unexpected keyword arguments: "
                f"{list(hyperparameters.keys())}"
            )
        super().__init__()
        self.input_variables = input_variables
        self.output_variables = output_variables
        self.sample_dim_name = sample_dim_name

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
    _OPTIONS_FILENAME = "options.yml"

    def __init__(
        self,
        sample_dim_name: str,
        input_variables: Iterable[str],
        output_variables: Iterable[str],
        weights: Optional[Mapping[str, Union[int, float, np.ndarray]]] = None,
        normalize_loss: bool = True,
    ):
        """Initialize the model.

        Loss is computed on normalized outputs only if `normalized_loss` is True
        (default). This allows you to provide weights that will be proportional
        to the importance of that feature within the loss. If `normalized_loss`
        is False, you should consider scaling your weights to decrease the importance
        of features that are orders of magnitude larger than other features.

        Args:
            sample_dim_name: name of the sample dimension in datasets used as
                inputs and outputs.
            input_variables: names of input variables
            output_variables: names of output variables
            weights: loss function weights, defined as a dict whose keys are
                variable names and values are either a scalar referring to the total
                weight of the variable, or a vector referring to the weight for each
                feature of the variable. Default is a total weight of 1
                for each variable.
            normalize_loss: if True (default), normalize outputs by their standard
                deviation before computing the loss function
        """
        super().__init__(sample_dim_name, input_variables, output_variables)
        self._model = None
        self.X_packer = ArrayPacker(
            sample_dim_name=sample_dim_name, pack_names=input_variables
        )
        self.y_packer = ArrayPacker(
            sample_dim_name=sample_dim_name, pack_names=output_variables
        )
        self.X_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        if weights is None:
            self.weights: Mapping[str, Union[int, float, np.ndarray]] = {}
        else:
            self.weights = weights
        self._normalize_loss = normalize_loss

    @property
    def model(self) -> tf.keras.Model:
        if self._model is None:
            raise RuntimeError("must call fit() for keras model to be available")
        return self._model

    def _fit_normalization(self, X: np.ndarray, y: np.ndarray):
        self.X_scaler.fit(X)
        self.y_scaler.fit(y)

    @abc.abstractmethod
    def get_model(self, n_features_in: int, n_features_out: int) -> tf.keras.Model:
        """Returns a Keras model to use as the underlying predictive model.

        Args:
            n_features_in: the number of input features
            n_features_out: the number of output features

        Returns:
            model: a Keras model whose input shape is [n_samples, n_features_in] and
                output shape is [n_samples, features_out]
        """
        pass

    def fit(self, batches: Sequence[xr.Dataset]) -> None:
        Xy = _XyArraySequence(self.X_packer, self.y_packer, batches)
        if self._model is None:
            X, y = Xy[0]
            n_features_in, n_features_out = X.shape[-1], y.shape[-1]
            self._fit_normalization(X, y)
            self._model = self.get_model(n_features_in, n_features_out)
        self.fit_array(Xy)

    def fit_array(self, X: Sequence[Tuple[np.ndarray, np.ndarray]]) -> None:
        return self.model.fit(X)

    def predict(self, X: xr.Dataset) -> xr.Dataset:
        return self.y_packer.to_dataset(self.predict_array(self.X_packer.to_array(X)))

    def predict_array(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def dump(self, path: str) -> None:
        with put_dir(path) as path:
            if self._model is not None:
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
            with open(os.path.join(path, self._OPTIONS_FILENAME), "w") as f:
                yaml.safe_dump({"normalize_loss": self._normalize_loss}, f)

    @property
    def loss(self):
        # putting this on a property method is needed so we can save and load models
        # using custom loss functions. If using a custom function, it must either
        # be named "custom_loss", as used in the load method below,
        # or it must be registered with keras as a custom object.
        # Do this by defining the function returned by the decorator as custom_loss.
        # See https://github.com/keras-team/keras/issues/5916 for more info
        std = self.y_scaler.std
        std[std == 0] = 1.0
        if not self._normalize_loss:
            std[:] = 1.0
        return get_weighted_mse(self.y_packer, std, **self.weights)

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
            with open(os.path.join(path, cls._OPTIONS_FILENAME), "r") as f:
                options = yaml.safe_load(f)
            obj = cls(
                X_packer.sample_dim_name,
                X_packer.pack_names,
                y_packer.pack_names,
                normalize_loss=options["normalize_loss"],
            )
            obj.X_packer = X_packer
            obj.y_packer = y_packer
            obj.X_scaler = X_scaler
            obj.y_scaler = y_scaler
            model_filename = os.path.join(path, cls._MODEL_FILENAME)
            if os.path.exists(model_filename):
                obj._model = tf.keras.models.load_model(
                    model_filename, custom_objects={"custom_loss": obj.loss}
                )
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
        weights: Optional[Mapping[str, Union[int, float, np.ndarray]]] = None,
        normalize_loss: bool = True,
        depth: int = 3,
        width: int = 16,
    ):
        """Initialize the DenseModel.

        Loss is computed on normalized outputs only if `normalized_loss` is True
        (default). This allows you to provide weights that will be proportional
        to the importance of that feature within the loss. If `normalized_loss`
        is False, you should consider scaling your weights to decrease the importance
        of features that are orders of magnitude larger than other features.

        Args:
            sample_dim_name: name of the sample dimension in datasets used as
                inputs and outputs.
            input_variables: names of input variables
            output_variables: names of output variables
            weights: loss function weights, defined as a dict whose keys are
                variable names and values are either a scalar referring to the total
                weight of the variable, or a vector referring to the weight for each
                feature of the variable. Default is a total weight of 1
                for each variable.
            normalize_loss: if True (default), normalize outputs by their standard
                deviation before computing the loss function
            depth: number of dense layers to use between the input and output layer.
                The number of hidden layers will be (depth - 1). Default is 3.
            width: number of neurons to use on layers between the input and output
                layer. Default is 16.
        """
        self._width = width
        self._depth = depth
        super().__init__(
            sample_dim_name,
            input_variables,
            output_variables,
            weights=weights,
            normalize_loss=normalize_loss,
        )

    def get_model(self, n_features_in: int, n_features_out: int) -> tf.keras.Model:
        inputs = tf.keras.Input(n_features_in)
        x = self.X_scaler.normalize_layer(inputs)
        for i in range(self._depth - 1):
            x = tf.keras.layers.Dense(
                self._width, activation=tf.keras.activations.relu
            )(x)
        x = tf.keras.layers.Dense(n_features_out, activation=tf.keras.activations.relu)(
            x
        )
        outputs = self.y_scaler.denormalize_layer(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer="sgd", loss=self.loss)
        return model


class DummyModel(PackedKerasModel):
    """
    A dummy neural network model for testing, which simply returns zeros for all
    model features
    """

    def __init__(
        self,
        sample_dim_name: str,
        input_variables: Iterable[str],
        output_variables: Iterable[str],
        normalize_loss: bool = False,
    ):
        """Initialize the DummyModel
        Args:
            sample_dim_name: name of the sample dimension in datasets used as
                inputs and outputs.
            input_variables: names of input variables
            output_variables: names of output variables
            normalize_loss: unused; this is here because the inherited load method
                initializes with this argument; TODO: overwrite load method to avoid
        """
        super().__init__(sample_dim_name, input_variables, output_variables)

    def get_model(self, n_features_in: int, n_features_out: int) -> tf.keras.Model:
        inputs = tf.keras.Input(n_features_in)
        outputs = tf.keras.layers.Lambda(lambda x: x * 0.0)(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile()
        return model
