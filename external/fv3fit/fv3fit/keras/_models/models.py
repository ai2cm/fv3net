from typing import Sequence, Tuple, Iterable, Mapping, Union, Optional, List, Any
import xarray as xr
import logging
import copy
import json
import tensorflow as tf
import tensorflow_addons as tfa
import tempfile
import dacite
import shutil
import dataclasses

from ..._shared.packer import ArrayPacker, unpack_matrix
from ..._shared.predictor import Predictor
from ..._shared import io, StackedBatches, stack_non_vertical
from ..._shared.config import (
    DenseHyperparameters,
    register_training_function,
    set_random_seed,
)
import numpy as np
import os
from ._filesystem import get_dir, put_dir
from ._sequences import _XyArraySequence, _ThreadedSequencePreLoader
from .normalizer import LayerStandardScaler
from .loss import get_weighted_mse, get_weighted_mae
from loaders.batches import Take, shuffle
import yaml

logger = logging.getLogger(__file__)

MODEL_DIRECTORY = "model_data"
KERAS_CHECKPOINT_PATH = "model_checkpoints"

# Description of the training loss progression over epochs
# Outer array indexes epoch, inner array indexes batch (if applicable)
EpochLossHistory = Sequence[Sequence[Union[float, int]]]
History = Mapping[str, EpochLossHistory]


@register_training_function("DenseModel", DenseHyperparameters)
def train_dense_model(
    input_variables: Iterable[str],
    output_variables: Iterable[str],
    hyperparameters: DenseHyperparameters,
    train_batches: Sequence[xr.Dataset],
    validation_batches: Sequence[xr.Dataset],
):
    set_random_seed(hyperparameters.random_seed)
    model = DenseModel("sample", input_variables, output_variables, hyperparameters)
    # TODO: make use of validation_batches, currently validation dataset is
    # passed through hyperparameters.fit_kwargs
    model.fit(train_batches)
    return model


@io.register("packed-keras")
class DenseModel(Predictor):
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
    _LOSS_OPTIONS = {"mse": get_weighted_mse, "mae": get_weighted_mae}
    _HISTORY_FILENAME = "training_history.json"

    def __init__(
        self,
        sample_dim_name: str,
        input_variables: Iterable[str],
        output_variables: Iterable[str],
        hyperparameters: DenseHyperparameters,
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
            hyperparameters: configuration of the dense model training
        """
        # store (duplicate) hyperparameters like this for ease of serialization
        self._hyperparameters = hyperparameters
        self._depth = hyperparameters.depth
        self._width = hyperparameters.width
        self._spectral_normalization = hyperparameters.spectral_normalization
        self._gaussian_noise = hyperparameters.gaussian_noise
        self._nonnegative_outputs = hyperparameters.nonnegative_outputs
        super().__init__(sample_dim_name, input_variables, output_variables)
        self._model = None
        self.X_packer = ArrayPacker(
            sample_dim_name=sample_dim_name, pack_names=input_variables
        )
        self.y_packer = ArrayPacker(
            sample_dim_name=sample_dim_name, pack_names=output_variables
        )
        self.X_scaler = LayerStandardScaler()
        self.y_scaler = LayerStandardScaler()
        self.train_history = {"loss": [], "val_loss": []}  # type: Mapping[str, List]
        if hyperparameters.weights is None:
            self.weights: Mapping[str, Union[int, float, np.ndarray]] = {}
        else:
            self.weights = hyperparameters.weights
        self._normalize_loss = hyperparameters.normalize_loss
        self._optimizer = hyperparameters.optimizer_config.instance
        self._loss = hyperparameters.loss
        self._epochs = hyperparameters.epochs
        if hyperparameters.kernel_regularizer_config is not None:
            regularizer = hyperparameters.kernel_regularizer_config.instance
        else:
            regularizer = None
        self._kernel_regularizer = regularizer
        self._save_model_checkpoints = hyperparameters.save_model_checkpoints
        if hyperparameters.save_model_checkpoints:
            self._checkpoint_path: Optional[
                tempfile.TemporaryDirectory
            ] = tempfile.TemporaryDirectory()
        else:
            self._checkpoint_path = None
        self._fit_kwargs = hyperparameters.fit_kwargs or {}
        self._random_seed = hyperparameters.random_seed

    @property
    def model(self) -> tf.keras.Model:
        if self._model is None:
            raise RuntimeError("must call fit() for keras model to be available")
        return self._model

    def _fit_normalization(self, X: np.ndarray, y: np.ndarray):
        self.X_scaler.fit(X)
        self.y_scaler.fit(y)

    def get_model(self, n_features_in: int, n_features_out: int) -> tf.keras.Model:
        inputs = tf.keras.Input(n_features_in)
        x = self.X_scaler.normalize_layer(inputs)
        for i in range(self._depth - 1):
            hidden_layer = tf.keras.layers.Dense(
                self._width,
                activation=tf.keras.activations.relu,
                kernel_regularizer=self._kernel_regularizer,
            )
            if self._spectral_normalization:
                hidden_layer = tfa.layers.SpectralNormalization(hidden_layer)
            if self._gaussian_noise > 0.0:
                x = tf.keras.layers.GaussianNoise(self._gaussian_noise)(x)
            x = hidden_layer(x)
        x = tf.keras.layers.Dense(n_features_out)(x)
        outputs = self.y_scaler.denormalize_layer(x)
        if self._nonnegative_outputs:
            outputs = tf.keras.layers.Activation(tf.keras.activations.relu)(outputs)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=self._optimizer, loss=self.loss)
        return model

    def fit(
        self,
        batches: Sequence[xr.Dataset],
        validation_dataset: Optional[xr.Dataset] = None,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        workers: Optional[int] = None,
        max_queue_size: Optional[int] = None,
        validation_samples: Optional[int] = None,
        use_last_batch_to_validate: Optional[bool] = None,
    ) -> None:
        """Fits a model using data in the batches sequence
        
        If batch_size is provided as a kwarg, the list of values is for each batch fit.
        e.g. {"loss":
            [[epoch0_batch0_loss, epoch0_batch1_loss],
            [epoch1_batch0_loss, epoch1_batch1_loss]]}
        If not batch_size is not provided, a single loss per epoch is recorded.
        e.g. {"loss": [[epoch0_loss], [epoch1_loss]]}
        
        Args:
            batches: sequence of unstacked datasets of predictor variables
            validation_dataset: optional validation dataset
            epochs: optional number of times through the batches to run when training.
                Defaults to 1.
            batch_size: actual batch_size to apply in gradient descent updates,
                independent of number of samples in each batch in batches; optional,
                uses number of samples in each batch if omitted
            workers: number of workers for parallelized loading of batches fed into
                training, defaults to serial loading (1 worker)
            max_queue_size: max number of batches to hold in the parallel loading queue.
                Defaults to 8.
            validation_samples: Option to specify number of samples to randomly draw
                from the validation dataset, so that we can use multiple timesteps for
                validation without having to load all the times into memory.
                Defaults to the equivalent of a single C48 timestep (13824).
            use_last_batch_to_validate: if True, use the last batch as a validation
                dataset, cannot be used with a non-None value for validation_dataset.
                Defaults to False.
        """

        fit_kwargs = copy.copy(self._fit_kwargs)
        fit_kwargs = _fill_default(fit_kwargs, batch_size, "batch_size", None)
        fit_kwargs = _fill_default(fit_kwargs, epochs, "epochs", self._epochs)
        fit_kwargs = _fill_default(fit_kwargs, workers, "workers", 1)
        fit_kwargs = _fill_default(fit_kwargs, max_queue_size, "max_queue_size", 8)
        fit_kwargs = _fill_default(
            fit_kwargs, validation_samples, "validation_samples", 13824
        )
        fit_kwargs = _fill_default(
            fit_kwargs, use_last_batch_to_validate, "use_last_batch_to_validate", False
        )
        stacked_batches = StackedBatches(
            batches, np.random.RandomState(self._random_seed)
        )
        Xy = _XyArraySequence(self.X_packer, self.y_packer, stacked_batches)
        if self._model is None:
            X, y = Xy[0]
            n_features_in, n_features_out = X.shape[-1], y.shape[-1]
            self._fit_normalization(X, y)
            self._model = self.get_model(n_features_in, n_features_out)

        validation_data: Optional[Tuple[np.ndarray, np.ndarray]]
        validation_dataset = (
            validation_dataset
            if validation_dataset is not None
            else fit_kwargs.pop("validation_dataset", None)
        )
        validation_samples = (
            validation_samples
            if validation_samples is not None
            else fit_kwargs.pop("validation_samples", 13824)
        )

        if use_last_batch_to_validate:
            if validation_dataset is not None:
                raise ValueError(
                    "cannot provide validation_dataset when "
                    "use_first_batch_to_validate is True"
                )
            X_val, y_val = Xy[-1]
            val_sample = np.random.choice(
                np.arange(X_val.shape[0]), validation_samples, replace=False
            )
            X_val = X_val[val_sample, :]
            y_val = y_val[val_sample, :]
            validation_data = (X_val, y_val)
            Xy = Take(Xy, len(Xy) - 1)  # type: ignore
        elif validation_dataset is not None:
            stacked_validation_dataset = stack_non_vertical(validation_dataset)
            X_val = self.X_packer.to_array(stacked_validation_dataset)
            y_val = self.y_packer.to_array(stacked_validation_dataset)
            val_sample = np.random.choice(
                np.arange(len(y_val)), validation_samples, replace=False
            )
            validation_data = X_val[val_sample], y_val[val_sample]
        else:
            validation_data = None

        return self._fit_loop(Xy, validation_data, **fit_kwargs,)

    def _fit_loop(
        self,
        Xy: Sequence[Tuple[np.ndarray, np.ndarray]],
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]],
        epochs: int,
        batch_size: Optional[int] = None,
        workers: int = 1,
        max_queue_size: int = 8,
        use_last_batch_to_validate: bool = False,
        last_batch_validation_fraction: float = 1.0,
        **fit_kwargs,
    ) -> None:

        for i_epoch in range(epochs):
            Xy = shuffle(Xy)
            if workers > 1:
                Xy = _ThreadedSequencePreLoader(
                    Xy, num_workers=workers, max_queue_size=max_queue_size
                )
            loss_over_batches, val_loss_over_batches = [], []
            for i_batch, (X, y) in enumerate(Xy):
                logger.info(
                    f"Fitting on batch {i_batch + 1} of {len(Xy)}, "
                    f"of epoch {i_epoch}..."
                )
                history = self.model.fit(
                    X,
                    y,
                    validation_data=validation_data,
                    batch_size=batch_size,
                    **fit_kwargs,
                )
                loss_over_batches += history.history["loss"]
                val_loss_over_batches += history.history.get("val_loss", [np.nan])
            self.train_history["loss"].append(loss_over_batches)
            self.train_history["val_loss"].append(val_loss_over_batches)
            if self._checkpoint_path:
                self.dump(os.path.join(self._checkpoint_path.name, f"epoch_{i_epoch}"))
                logger.info(
                    f"Saved model checkpoint after epoch {i_epoch} "
                    f"to {self._checkpoint_path}"
                )

    def predict(self, X: xr.Dataset) -> xr.Dataset:
        sample_coord = X[self.sample_dim_name]
        ds_pred = self.y_packer.to_dataset(
            self.predict_array(self.X_packer.to_array(X))
        )
        return ds_pred.assign_coords({self.sample_dim_name: sample_coord})

    def predict_array(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def dump(self, path: str) -> None:
        dir_ = os.path.join(path, MODEL_DIRECTORY)
        with put_dir(dir_) as path:
            if self._model is not None:
                model_filename = os.path.join(path, self._MODEL_FILENAME)
                self.model.save(model_filename)
            if self._checkpoint_path is not None:
                shutil.copytree(
                    self._checkpoint_path.name,
                    os.path.join(path, KERAS_CHECKPOINT_PATH),
                )
            with open(os.path.join(path, self._X_PACKER_FILENAME), "w") as f:
                self.X_packer.dump(f)
            with open(os.path.join(path, self._Y_PACKER_FILENAME), "w") as f:
                self.y_packer.dump(f)
            with open(os.path.join(path, self._X_SCALER_FILENAME), "wb") as f_binary:
                self.X_scaler.dump(f_binary)
            with open(os.path.join(path, self._Y_SCALER_FILENAME), "wb") as f_binary:
                self.y_scaler.dump(f_binary)
            with open(os.path.join(path, self._OPTIONS_FILENAME), "w") as f:
                # TODO: remove this hack when we aren't
                # putting validation data in fit_kwargs
                options = dataclasses.asdict(self._hyperparameters)
                fit_kwargs = options.get("fit_kwargs", {})
                if fit_kwargs is None:  # it is sometimes present with a value of None
                    fit_kwargs = {}
                if "validation_dataset" in fit_kwargs:
                    fit_kwargs.pop("validation_dataset")
                yaml.safe_dump(options, f)
            with open(os.path.join(path, self._HISTORY_FILENAME), "w") as f:
                json.dump(self.train_history, f)

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
        if self._loss in self._LOSS_OPTIONS:
            loss_getter = self._LOSS_OPTIONS[self._loss]
            return loss_getter(self.y_packer, std, **self.weights)
        else:
            raise ValueError(
                f"Invalid loss {self._loss} provided. "
                f"Allowed loss functions are {list(self._LOSS_OPTIONS.keys())}."
            )

    @classmethod
    def load(cls, path: str) -> "DenseModel":
        dir_ = os.path.join(path, MODEL_DIRECTORY)
        with get_dir(dir_) as path:
            with open(os.path.join(path, cls._X_PACKER_FILENAME), "r") as f:
                X_packer = ArrayPacker.load(f)
            with open(os.path.join(path, cls._Y_PACKER_FILENAME), "r") as f:
                y_packer = ArrayPacker.load(f)
            with open(os.path.join(path, cls._X_SCALER_FILENAME), "rb") as f_binary:
                X_scaler = LayerStandardScaler.load(f_binary)
            with open(os.path.join(path, cls._Y_SCALER_FILENAME), "rb") as f_binary:
                y_scaler = LayerStandardScaler.load(f_binary)
            with open(os.path.join(path, cls._OPTIONS_FILENAME), "r") as f:
                options = yaml.safe_load(f)
            hyperparameters = dacite.from_dict(
                data_class=DenseHyperparameters, data=options
            )

            obj = cls(
                X_packer.sample_dim_name,
                X_packer.pack_names,
                y_packer.pack_names,
                hyperparameters,
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
            history_filename = os.path.join(path, cls._HISTORY_FILENAME)
            if os.path.exists(history_filename):
                with open(os.path.join(path, cls._HISTORY_FILENAME), "r") as f:
                    obj.train_history = json.load(f)
            return obj

    def jacobian(self, base_state: Optional[xr.Dataset] = None) -> xr.Dataset:
        """Compute the jacobian of the NN around a base state

        Args:
            base_state: a single sample of input data. If not passed, then
                the mean of the input data stored in the X_scaler will be used.

        Returns:
            The jacobian matrix as a Dataset

        """
        if base_state is None:
            if self.X_scaler.mean is not None:
                mean_expanded = self.X_packer.to_dataset(
                    self.X_scaler.mean[np.newaxis, :]
                )
            else:
                raise ValueError("X_scaler needs to be fit first.")
        else:
            mean_expanded = base_state.expand_dims(self.sample_dim_name)

        mean_tf = tf.convert_to_tensor(self.X_packer.to_array(mean_expanded))
        with tf.GradientTape() as g:
            g.watch(mean_tf)
            y = self.model(mean_tf)

        J = g.jacobian(y, mean_tf)[0, :, 0, :].numpy()
        return unpack_matrix(self.X_packer, self.y_packer, J)


def _fill_default(kwargs: dict, arg: Optional[Any], key: str, default: Any):
    if key not in kwargs:
        if arg is None:
            kwargs[key] = default
        else:
            kwargs[key] = arg
    else:
        if arg is not None and arg != kwargs[key]:
            raise ValueError(
                f"Different values for fit kwarg {key} were provided in both "
                "fit args and fit_kwargs dict."
            )
    return kwargs
