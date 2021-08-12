import dataclasses
from typing import Iterable, Hashable, Mapping, Optional, Sequence, Union
from fv3fit._shared.config import (
    OptimizerConfig,
    RegularizerConfig,
    register_training_function,
)
from fv3fit._shared.stacking import SAMPLE_DIM_NAME, StackedBatches
import tensorflow as tf
import xarray as xr
import os
import tempfile
from ..._shared import ArrayPacker
from ._sequences import _ThreadedSequencePreLoader
from .packer import get_unpack_layer
from .normalizer import LayerStandardScaler
import numpy as np
from .recurrent import get_input_vector, PureKerasModel
from loaders.batches import shuffle
import logging

logger = logging.getLogger(__file__)

LV = 2.5e6  # Latent heat of evaporation [J/kg]
WATER_DENSITY = 997  # kg/m^3
CPD = 1004.6  # Specific heat capacity of dry air at constant pressure [J/kg/deg]
GRAVITY = 9.80665  # m /s2
KG_M2_TO_MM = 1000.0 / WATER_DENSITY  # mm/m / (kg/m^3) = mm / (kg/m^2)


def integrate_precip(args):
    dQ2, delp = args  # layers seem to take in a single argument
    return tf.math.scalar_mul(
        KG_M2_TO_MM / GRAVITY, tf.math.reduce_sum(tf.math.multiply(dQ2, delp), axis=-1)
    )


def evaporative_heating(dQ2):
    return tf.math.scalar_mul(tf.constant(LV / CPD), dQ2)


def multiply_loss_by_factor(original_loss, factor):
    def loss(y_true, y_pred):
        return tf.math.multiply(factor, original_loss(y_true, y_pred))

    return loss


def get_losses(output_variables: Sequence[str], output_packer, output_scaler):
    std = output_packer.to_dataset(output_scaler.std[None, :])

    # we want each output to contribute equally, so for a
    # mean squared error loss we need to normalize by the variance of
    # each one
    # we want layers to have importance which is proportional to how much
    # variance there is in that layer, so the variance we use should be
    # constant across layers
    # we need to compute the variance independently for each layer and then
    # average them, so that we don't include variance in the mean profile
    # we use a 1/N scale for each one, so the total expected loss is 1.0 if
    # we have zero skill

    n_outputs = len(output_packer.pack_names)
    loss_list = []
    for name in output_variables:
        factor = tf.constant(
            1.0 / n_outputs / np.mean(std[name].values ** 2), dtype=tf.float32
        )
        loss_list.append(multiply_loss_by_factor(tf.losses.mse, factor))

    return loss_list


def get_output_metadata(packer, sample_dim_name):
    """
    Retrieve xarray metadata for a packer's values, assuming arrays are [sample(, z)].
    """
    metadata = []
    for name in packer.pack_names:
        n_features = packer.feature_counts[name]
        if n_features == 1:
            dims = [sample_dim_name]
        else:
            dims = [sample_dim_name, "z"]
        metadata.append({"dims": dims, "units": "unknown"})
    return tuple(metadata)


@dataclasses.dataclass
class PrecipitativeHyperparameters:
    """
    Configuration for training a neural network with a closed,
    optimized precipitation budget.

    Uses only the first batch of any validation data it is given.

    Args:
        weights: loss function weights, defined as a dict whose keys are
            variable names and values are either a scalar referring to the total
            weight of the variable. Default is a total weight of 1
            for each variable.
        normalize_loss: if True (default), normalize outputs by their standard
            deviation before computing the loss function
        optimizer_config: selection of algorithm to be used in gradient descent
        kernel_regularizer_config: selection of regularizer for hidden dense layer
            weights, by default no regularization is applied
        dynamics_regularizer_config: selection of regularizer for unconstrainted
            (non-flux based) tendency output, by default no regularization is applied
        depth: number of dense layers to use between the input and output layer.
            The number of hidden layers will be (depth - 1)
        width: number of neurons to use on layers between the input and output layer
        gaussian_noise: how much gaussian noise to add before each Dense layer,
            apart from the output layer
        loss: loss function to use, should be 'mse' or 'mae'
        spectral_normalization: whether to apply spectral normalization to hidden layers
        save_model_checkpoints: if True, save one model per epoch when
            dumping, under a 'model_checkpoints' subdirectory
        nonnegative_outputs: if True, add a ReLU activation layer as the last layer
            after output denormalization layer to ensure outputs are always >=0
            Defaults to False.
        workers: number of workers for parallelized loading of batches fed into
            training, defaults to serial loading (1 worker)
        max_queue_size: max number of batches to hold in the parallel loading queue.
            Defaults to 8.
        keras_batch_size: actual batch_size to apply in gradient descent updates,
            independent of number of samples in each batch in batches; optional,
            uses 32 if omitted
    """

    weights: Optional[Mapping[str, Union[int, float]]] = None
    normalize_loss: bool = True
    optimizer_config: OptimizerConfig = dataclasses.field(
        default_factory=lambda: OptimizerConfig("Adam")
    )
    kernel_regularizer_config: RegularizerConfig = dataclasses.field(
        default_factory=lambda: RegularizerConfig("none")
    )
    dynamics_regularizer_config: RegularizerConfig = dataclasses.field(
        default_factory=lambda: RegularizerConfig("none")
    )
    depth: int = 3
    width: int = 16
    epochs: int = 3
    gaussian_noise: float = 0.0
    loss: str = "mse"
    spectral_normalization: bool = False
    save_model_checkpoints: bool = False
    nonnegative_outputs: bool = False
    workers: int = 1
    max_queue_size: int = 8
    keras_batch_size: int = 32


@register_training_function("precipitative", PrecipitativeHyperparameters)
def train_precipitative_model(
    input_variables: Iterable[str],
    output_variables: Iterable[str],
    hyperparameters: PrecipitativeHyperparameters,
    train_batches: Sequence[xr.Dataset],
    validation_batches: Sequence[xr.Dataset],
):
    random_state = np.random.RandomState(np.random.get_state()[1][0])
    stacked_train_batches = StackedBatches(train_batches, random_state)
    stacked_validation_batches = StackedBatches(validation_batches, random_state)
    training_obj = PrecipitativeModel(
        input_variables=input_variables,
        output_variables=output_variables,
        epochs=hyperparameters.epochs,
        workers=hyperparameters.workers,
        max_queue_size=hyperparameters.max_queue_size,
        depth=hyperparameters.depth,
        width=hyperparameters.width,
        train_batch_size=hyperparameters.keras_batch_size,
        optimizer=hyperparameters.optimizer_config.instance,
        kernel_regularizer=hyperparameters.kernel_regularizer_config.instance,
        dynamics_regularizer=hyperparameters.dynamics_regularizer_config.instance,
        save_model_checkpoints=hyperparameters.save_model_checkpoints,
    )
    training_obj.fit_statistics(stacked_train_batches[0])
    if len(stacked_validation_batches) > 0:
        val_batch = stacked_validation_batches[0]
    else:
        val_batch = None
    training_obj.fit(stacked_train_batches, validation_batch=val_batch)
    return training_obj.predictor


class PrecipitativeModel:

    _DELP_NAME = "pressure_thickness_of_atmospheric_layer"
    _T_NAME = "air_temperature"
    _Q_NAME = "specific_humidity"
    _PRECIP_NAME = "total_precipitation_rate"
    _T_TENDENCY_NAME = "dQ1"
    _Q_TENDENCY_NAME = "dQ2"

    def __init__(
        self,
        input_variables: Iterable[Hashable],
        output_variables: Iterable[Hashable],
        epochs: int = 1,
        workers: int = 1,
        max_queue_size: int = 8,
        depth: int = 3,
        width: int = 16,
        train_batch_size: Optional[int] = None,
        optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
        kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
        dynamics_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
        save_model_checkpoints: bool = False,
    ):
        self._check_missing_names(input_variables, output_variables)
        self.sample_dim_name = SAMPLE_DIM_NAME
        self.output_variables = self._order_outputs(output_variables)
        self.input_variables = input_variables
        self.input_packer = ArrayPacker(self.sample_dim_name, input_variables)
        self.prognostic_packer = ArrayPacker(
            self.sample_dim_name, [self._T_NAME, self._Q_NAME]
        )
        self.humidity_packer = ArrayPacker(self.sample_dim_name, [self._Q_NAME])
        output_without_precip = (n for n in output_variables if n != self._PRECIP_NAME)
        self.output_packer = ArrayPacker(self.sample_dim_name, self.output_variables)
        self.output_without_precip_packer = ArrayPacker(
            self.sample_dim_name, output_without_precip
        )
        self.input_scaler = LayerStandardScaler()
        self.prognostic_scaler = LayerStandardScaler()
        self.output_scaler = LayerStandardScaler()
        self.output_without_precip_scaler = LayerStandardScaler()
        self.humidity_scaler = LayerStandardScaler()
        self._train_model: Optional[tf.keras.Model] = None
        self._predict_model: Optional[tf.keras.Model] = None
        self._epochs = epochs
        self._workers = workers
        self._max_queue_size = max_queue_size
        self._statistics_are_fit = False
        self._depth = depth
        self._width = width
        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam()
        self._optimizer = optimizer
        self._kernel_regularizer = kernel_regularizer
        self._save_model_checkpoints = save_model_checkpoints
        if save_model_checkpoints:
            self._checkpoint_path: Optional[
                tempfile.TemporaryDirectory
            ] = tempfile.TemporaryDirectory()
        else:
            self._checkpoint_path = None
        self._train_batch_size = train_batch_size
        self.train_history = {"loss": [], "val_loss": []}
        self._dynamics_regularizer = dynamics_regularizer

    def _check_missing_names(self, input_variables, output_variables):
        missing_names = set((self._T_NAME, self._Q_NAME, self._DELP_NAME)).difference(
            input_variables
        )
        if len(missing_names) > 0:
            raise ValueError(
                "input_variables for PrecipitativeModel requires "
                f"the following missing variables: {missing_names}"
            )
        missing_names = set(
            (self._T_TENDENCY_NAME, self._Q_TENDENCY_NAME, self._PRECIP_NAME)
        ).difference(output_variables)
        if len(missing_names) > 0:
            raise ValueError(
                "output_variables for PrecipitativeModel requires "
                f"the following missing variables: {missing_names}"
            )

    def _order_outputs(self, output_variables: Sequence[str]):
        """
        To keep some of the implementation sane, we hard-code that the first
        outputs are temperature, humidity, and precipitation, and all others
        come after.
        """
        return_list = [self._T_TENDENCY_NAME, self._Q_TENDENCY_NAME, self._PRECIP_NAME]
        for name in output_variables:
            if name not in return_list:
                return_list.append(name)
        assert len(return_list) == len(output_variables)
        return tuple(return_list)

    def fit_statistics(self, X: xr.Dataset):
        """
        Given a dataset with [sample, z] and [sample] arrays, fit the
        scalers and packers.
        """
        inputs = self.input_packer.to_array(X)
        self.input_scaler.fit(inputs)
        state = self.prognostic_packer.to_array(X)
        self.prognostic_scaler.fit(state)
        outputs = self.output_packer.to_array(X)
        self.output_scaler.fit(outputs)
        outputs_without_precip = self.output_without_precip_packer.to_array(X)
        self.output_without_precip_scaler.fit(outputs_without_precip)
        humidity = self.humidity_packer.to_array(X)
        self.humidity_scaler.fit(humidity)
        assert tuple(self.output_packer.pack_names) == tuple(self.output_variables), (
            self.output_packer.pack_names,
            self.output_variables,
        )
        self._statistics_are_fit = True

    def _build_model(self):
        input_layers, input_vector = get_input_vector(
            self.input_packer, scaler=self.input_scaler, n_window=None, series=False
        )
        # q = input_layers[self.input_variables.index(self._Q_NAME)]
        delp = input_layers[self.input_variables.index(self._DELP_NAME)]

        x = input_vector
        for i in range(self._depth - 1):
            hidden_layer = tf.keras.layers.Dense(
                self._width,
                activation=tf.keras.activations.relu,
                kernel_regularizer=self._kernel_regularizer,
                name=f"hidden_layer_{i}",
            )
            x = hidden_layer(x)
        output_features = sum(self.output_without_precip_packer.feature_counts.values())
        output_vector = tf.keras.layers.Dense(
            output_features,
            activation="linear",
            activity_regularizer=self._dynamics_regularizer,
            name="dense_output",
        )(x)
        denormalized_output = self.output_without_precip_scaler.denormalize_layer(
            output_vector
        )
        unpacked_output = get_unpack_layer(
            self.output_without_precip_packer, feature_dim=1
        )(denormalized_output)
        T_tendency = unpacked_output[0]
        q_tendency = unpacked_output[1]

        column_precip_vector = tf.keras.layers.Dense(
            self.humidity_packer.feature_counts[self._Q_NAME], activation="linear"
        )(x)
        column_precip = self.humidity_scaler.denormalize_layer(column_precip_vector)
        column_precip = tf.keras.activations.relu(column_precip)
        column_heating = tf.keras.layers.Lambda(evaporative_heating)(column_precip)
        T_tendency = tf.keras.layers.Add(name="T_tendency")(
            [T_tendency, column_heating]
        )
        q_tendency = tf.keras.layers.Subtract(name="q_tendency")(
            [q_tendency, column_precip]
        )
        surface_precip = tf.keras.layers.Lambda(
            integrate_precip, name="surface_precip"
        )([column_precip, delp])
        train_model = tf.keras.Model(
            inputs=input_layers,
            outputs=(T_tendency, q_tendency, surface_precip)
            + tuple(unpacked_output[2:]),
        )
        train_model.compile(
            optimizer=self._optimizer,
            loss=get_losses(
                self.output_variables, self.output_packer, self.output_scaler
            ),
        )
        # need a separate model for this so we don't have to
        # serialize the custom loss functions
        predict_model = tf.keras.Model(
            inputs=input_layers,
            outputs=(T_tendency, q_tendency, surface_precip)
            + tuple(unpacked_output[2:]),
        )
        return train_model, predict_model

    def fit(self, batches: Sequence[xr.Dataset], validation_batch: xr.Dataset) -> None:
        """Fits a model using data in the batches sequence.
        """
        if not self._statistics_are_fit:
            self.fit_statistics(batches[0])
        if self._train_model is None:
            self._train_model, self._predict_model = self._build_model()
        return self._fit_loop(batches, validation_batch)

    def _fit_loop(
        self, batches: Sequence[xr.Dataset], validation_batch: xr.Dataset
    ) -> None:
        if validation_batch is None:
            validation_data = None
        else:
            validation_data = (
                tuple(validation_batch[name].values for name in self.input_variables),
                tuple(validation_batch[name].values for name in self.output_variables),
            )
        for i_epoch in range(self._epochs):
            epoch_batches = shuffle(batches)
            if self._workers > 1:
                epoch_batches = _ThreadedSequencePreLoader(
                    epoch_batches,
                    num_workers=self._workers,
                    max_queue_size=self._max_queue_size,
                )
            loss_over_batches, val_loss_over_batches = [], []
            for i_batch, batch in enumerate(epoch_batches):
                logger.info(
                    f"Fitting on batch {i_batch + 1} of {len(epoch_batches)}, "
                    f"of epoch {i_epoch}..."
                )
                history = self._train_model.fit(
                    x=tuple(batch[name].values for name in self.input_variables),
                    y=tuple(batch[name].values for name in self.output_variables),
                    batch_size=self._train_batch_size,
                    validation_data=validation_data,
                )
                loss_over_batches += history.history["loss"]
                val_loss_over_batches += history.history.get("val_loss", [np.nan])
            self.train_history["loss"].append(loss_over_batches)
            self.train_history["val_loss"].append(val_loss_over_batches)
            if self._checkpoint_path:
                self.dump(os.path.join(self._checkpoint_path, f"epoch_{i_epoch}"))
                logger.info(
                    f"Saved model checkpoint after epoch {i_epoch} "
                    f"to {self._checkpoint_path}"
                )

    @property
    def predictor(self) -> PureKerasModel:
        return PureKerasModel(
            self.sample_dim_name,
            self.input_variables,
            self.output_variables,
            get_output_metadata(self.output_packer, self.sample_dim_name),
            self._predict_model,
        )
