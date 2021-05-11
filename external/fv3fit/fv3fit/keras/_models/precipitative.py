from typing import Dict, Any, Iterable, Hashable, Optional, Sequence
import tensorflow as tf
import xarray as xr
import os
from ..._shared import io, ArrayPacker
from ._sequences import _ThreadedSequencePreLoader
from .packer import get_unpack_layer
from .normalizer import LayerStandardScaler
import numpy as np
from .recurrent import get_input_vector, PureKerasModel
from ..._shared.predictor import Estimator
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


def get_losses(output_packer, output_scaler):
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
    for name in output_packer.pack_names:
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


@io.register("precipitative")
class PrecipitativeModel(Estimator):

    _MODEL_FILENAME = "model.tf"
    _CONFIG_FILENAME = "config.yaml"
    custom_objects: Dict[str, Any] = {}

    _DELP_NAME = "pressure_thickness_of_atmospheric_layer"
    _T_NAME = "air_temperature"
    _Q_NAME = "specific_humidity"
    _PRECIP_NAME = "total_precipitation_rate"
    _T_TENDENCY_NAME = "dQ1"
    _Q_TENDENCY_NAME = "dQ2"

    def __init__(
        self,
        sample_dim_name: str,
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
        dynamics_l2_regularization: float = 0.0,
        checkpoint_path: Optional[str] = None,
        fit_kwargs: Optional[dict] = None,
    ):
        """Initialize the predictor
        
        Args:
            sample_dim_name: name of sample dimension
            input_variables: names of input variables
            output_variables: names of output variables
        """
        self._check_missing_names(input_variables, output_variables)
        super().__init__(sample_dim_name, input_variables, output_variables)
        self.sample_dim_name = sample_dim_name
        self.output_variables = self._order_outputs(output_variables)
        self.input_variables = input_variables
        self.input_packer = ArrayPacker(sample_dim_name, input_variables)
        self.prognostic_packer = ArrayPacker(
            sample_dim_name, [self._T_NAME, self._Q_NAME]
        )
        self.humidity_packer = ArrayPacker(sample_dim_name, [self._Q_NAME])
        output_without_precip = (n for n in output_variables if n != self._PRECIP_NAME)
        self.output_packer = ArrayPacker(sample_dim_name, self.output_variables)
        self.output_without_precip_packer = ArrayPacker(
            sample_dim_name, output_without_precip
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
        self._checkpoint_path = checkpoint_path
        self._train_batch_size = train_batch_size
        if fit_kwargs is None:
            fit_kwargs = {}
        self._fit_kwargs = fit_kwargs
        self.train_history = {"loss": [], "val_loss": []}
        self._dynamics_regularizer = tf.keras.regularizers.l2(
            dynamics_l2_regularization
        )

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
        state = self.prognostic_packer.to_array(X)
        outputs = self.output_packer.to_array(X)
        outputs_without_precip = self.output_without_precip_packer.to_array(X)
        humidity = self.humidity_packer.to_array(X)
        self.input_scaler.fit(inputs)
        self.prognostic_scaler.fit(state)
        self.output_scaler.fit(outputs)
        self.output_without_precip_scaler.fit(outputs_without_precip)
        self.humidity_scaler.fit(humidity)
        self._statistics_are_fit = True

    def _build_model(self):
        input_layers, input_vector = get_input_vector(
            self.input_packer, scaler=self.input_scaler, n_window=None, series=False
        )
        # q = input_layers[self.input_variables.index(self._Q_NAME)]
        delp = input_layers[self.input_variables.index(self._DELP_NAME)]

        x = input_vector
        for _ in range(self._depth - 1):
            hidden_layer = tf.keras.layers.Dense(
                self._width,
                activation=tf.keras.activations.relu,
                kernel_regularizer=self._kernel_regularizer,
            )
            x = hidden_layer(x)
        output_features = sum(self.output_without_precip_packer.feature_counts.values())
        output_vector = tf.keras.layers.Dense(
            output_features,
            activation="linear",
            activity_regularizer=self.dynamics_regularizer,
        )(x)
        denormalized_output = self.output_without_precip_scaler.denormalize_layer(
            output_vector
        )
        unpacked_output = get_unpack_layer(
            self.output_without_precip_packer, feature_dim=1
        )(denormalized_output)
        q_tendency = unpacked_output[0]
        T_tendency = unpacked_output[1]

        column_precip_vector = tf.keras.layers.Dense(
            self.humidity_packer.feature_counts[self._Q_NAME], activation="relu"
        )(x)
        column_precip = self.humidity_scaler.denormalize_layer(column_precip_vector)
        column_heating = tf.keras.layers.Lambda(evaporative_heating)(column_precip)
        T_tendency = tf.keras.layers.Add()([T_tendency, column_heating])
        q_tendency = tf.keras.layers.Subtract()([q_tendency, column_precip])
        surface_precip = tf.keras.layers.Lambda(integrate_precip)([column_precip, delp])
        train_model = tf.keras.Model(
            inputs=input_layers,
            outputs=(T_tendency, q_tendency, surface_precip)
            + tuple(unpacked_output[2:]),
        )
        train_model.compile(
            optimizer=self._optimizer,
            loss=get_losses(self.output_packer, self.output_scaler),
        )
        # need a separate model for this so we don't have to
        # serialize the custom loss functions
        predict_model = tf.keras.Model(
            inputs=input_layers,
            outputs=(T_tendency, q_tendency, surface_precip)
            + tuple(unpacked_output[2:]),
        )
        return train_model, predict_model

    def fit(self, batches: Sequence[xr.Dataset],) -> None:
        """Fits a model using data in the batches sequence.
        """
        if not self._statistics_are_fit:
            self.fit_statistics(batches[0])
        if self._train_model is None:
            self._train_model, self._predict_model = self._build_model()
        return self._fit_loop(batches)

    def _fit_loop(self, batches: Sequence[xr.Dataset],) -> None:
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
                    **self._fit_kwargs,
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
    def predictor(self):
        return PureKerasModel(
            self.sample_dim_name,
            self.input_variables,
            self.output_variables,
            get_output_metadata(self.output_packer, self.sample_dim_name),
            self._predict_model,
        )

    @classmethod
    def load(cls, path: str) -> "PureKerasModel":
        """Load a serialized model from a directory."""
        return PureKerasModel.load(path)

    def predict(self, X: xr.Dataset) -> xr.Dataset:
        return self.predictor.predict(X)

    def dump(self, path: str) -> None:
        self.predictor.dump(path)
