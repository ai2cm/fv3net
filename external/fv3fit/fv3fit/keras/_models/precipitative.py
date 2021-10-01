import dataclasses
from typing import List, Optional, Sequence, Tuple, Set, Mapping, Union
from fv3fit._shared.config import (
    OptimizerConfig,
    RegularizerConfig,
    register_training_function,
)
from fv3fit._shared.scaler import StandardScaler
from fv3fit._shared.stacking import SAMPLE_DIM_NAME, StackedBatches
import tensorflow as tf
import xarray as xr
from ..._shared import ArrayPacker
from ..._shared.config import Hyperparameters
from ._sequences import _XyMultiArraySequence
from .packer import get_unpack_layer
from .normalizer import LayerStandardScaler
from .shared import DenseNetworkConfig, TrainingLoopConfig
import numpy as np
from fv3fit.keras._models.shared import get_input_vector, PureKerasModel
import logging


logger = logging.getLogger(__file__)

LV = 2.5e6  # Latent heat of evaporation [J/kg]
WATER_DENSITY = 997  # kg/m^3
CPD = 1004.6  # Specific heat capacity of dry air at constant pressure [J/kg/deg]
GRAVITY = 9.80665  # m /s2

DELP_NAME = "pressure_thickness_of_atmospheric_layer"
T_NAME = "air_temperature"
Q_NAME = "specific_humidity"
PRECIP_NAME = "total_precipitation_rate"
PHYS_PRECIP_NAME = "physics_precip"
T_TENDENCY_NAME = "dQ1"
Q_TENDENCY_NAME = "dQ2"


def integrate_precip(args):
    dQ2, delp = args  # layers seem to take in a single argument
    # output should be kg/m^2/s
    return tf.math.scalar_mul(
        # dQ2 * delp = s-1 * Pa = s^-1 * kg m-1 s-2 = kg m-1 s-3
        # dQ2 * delp / g = kg m-1 s-3 / (m s-2) = kg/m^2/s
        tf.constant(-1.0 / GRAVITY, dtype=dQ2.dtype),
        tf.math.reduce_sum(tf.math.multiply(dQ2, delp), axis=-1),
    )


def condensational_heating(dQ2):
    """
    Args:
        dQ2: rate of change in moisture in kg/kg/s, negative corresponds
            to condensation
    
    Returns:
        heating rate in degK/s
    """
    return tf.math.scalar_mul(tf.constant(-LV / CPD, dtype=dQ2.dtype), dQ2)


def multiply_loss_by_factor(original_loss, factor):
    def loss(y_true, y_pred):
        return tf.math.scalar_mul(factor, original_loss(y_true, y_pred))

    return loss


def get_losses(
    output_variables: Sequence[str],
    output_packer: ArrayPacker,
    output_scaler: StandardScaler,
    loss_type="mse",
    weights={},
) -> Sequence[tf.keras.losses.Loss]:
    """
    Retrieve normalized losses for a sequence of output variables.

    Returned losses are such that a predictor with no skill making random
    predictions should return a loss of about 1. For MSE loss, each
    column is normalized by the variance of the data after subtracting
    the vertical mean profile (if relevant), while for MAE loss the
    columns are normalized by the standard deviation of the data after
    subtracting the vertical mean profile. This variance or standard
    deviation is read from the given `output_scaler`.

    After this scaling, each loss is also divided by the total number
    of outputs, so that the total loss sums roughly to 1 for a model
    with no skill.

    Args:
        output_variables: names of outputs in order
        output_packer: packer which includes all of the output variables,
            in any order (having more variables is fine)
        output_scaler: scaler which has been fit to all output variables
        loss_type: can be "mse" or "mae"
    
    Returns:
        loss_list: loss functions for output variables in order
    """
    if output_scaler.std is None:
        raise ValueError("output_scaler must be fit before passing it to this function")
    std = output_packer.to_dataset(output_scaler.std[None, :])

    # we want each output to contribute equally, so for a
    # mean squared error loss we need to normalize by the variance of
    # each one
    # here we want layers to have importance which is proportional to how much
    # variance there is in that layer, so the variance we use should be
    # constant across layers
    # we need to compute the variance independently for each layer and then
    # average them, so that we don't include variance in the mean profile
    # we use a 1/N scale for each one, so the total expected loss is 1.0 if
    # we have zero skill

    n_outputs = len(output_variables)
    loss_list = []
    for name in output_variables:
        weight = weights.get(name, 1.0)
        if loss_type == "mse":
            factor = tf.constant(
                weight / n_outputs / np.mean(std[name].values ** 2), dtype=tf.float32
            )
            loss = multiply_loss_by_factor(tf.losses.mse, factor)
        elif loss_type == "mae":
            factor = tf.constant(
                weight / n_outputs / np.mean(std[name].values), dtype=tf.float32
            )
            loss = multiply_loss_by_factor(tf.losses.mae, factor)
        else:
            raise NotImplementedError(f"loss_type {loss_type} is not implemented")
        loss_list.append(loss)

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
class PrecipitativeHyperparameters(Hyperparameters):
    """
    Configuration for training a neural network with a closed,
    optimized precipitation budget.

    Uses only the first batch of any validation data it is given.

    The basic NN architecture is the same as in previous "Dense" NN models used in the
    N2F experiments. The inputs are used to predict dQ1 and dQ2
    (full nudging tendencies).

    The differences are just that:
        - The total precipitation rate is also an output
        - Total precipitation rate is the sum of a column-integrated column
          precipitation internal layer, and physics precipitation
        - dQ2 is specified as the sum of the column precipitation internal layer
          and a column moistening residual internal layer
        - dQ1 is the sum of the column precipitation internal layer converted to
          heating and a column heating residual internal layer
    
    So the behavior is such that precipitation is predicted and the ML portion of the
    precipitation is also a portion of the dQ1 and dQ2 outputs.

    There is also a flag couple_precip_to_dQ1_dQ2 to turn off the column linkage
    (when turned off this is the “dense-like” version, i.e., just predict surface
    precipitation, dQ1, and dQ2 separately without constraints on their relationship).
    In this case the model behavior is just the same as previous N2F NNs, except that
    there are more input features and output features).

    Args:
        additional_input_variables: if given, used as input variables in
            addition to the default inputs (air_temperature, specific_humidity,
            physics_precip, and pressure_thickness_of_atmospheric_layer)
        optimizer_config: selection of algorithm to be used in gradient descent
        dense_network: configuration for dense component of network
        residual_regularizer_config: selection of regularizer for unconstrainted
            (non-flux based) tendency output, by default no regularization is applied
        training_loop: configuration of training loop
        loss: loss function to use, should be 'mse' or 'mae'
        couple_precip_to_dQ1_dQ2: if False, try to recover behavior of Dense model type
            by not adding "precipitative" terms to dQ1 and dQ2
    """

    additional_input_variables: List[str] = dataclasses.field(default_factory=list)
    optimizer_config: OptimizerConfig = dataclasses.field(
        default_factory=lambda: OptimizerConfig("Adam")
    )
    dense_network: DenseNetworkConfig = dataclasses.field(
        default_factory=lambda: DenseNetworkConfig(width=16)
    )
    residual_regularizer_config: RegularizerConfig = dataclasses.field(
        default_factory=lambda: RegularizerConfig("none")
    )
    training_loop: TrainingLoopConfig = dataclasses.field(
        default_factory=TrainingLoopConfig
    )
    loss: str = "mse"
    couple_precip_to_dQ1_dQ2: bool = True
    weights: Optional[Mapping[str, Union[int, float]]] = None

    @property
    def variables(self) -> Set[str]:
        return set(
            [
                T_NAME,
                Q_NAME,
                DELP_NAME,
                PHYS_PRECIP_NAME,
                T_TENDENCY_NAME,
                Q_TENDENCY_NAME,
                PRECIP_NAME,
            ]
        ).union(self.additional_input_variables)


@register_training_function("precipitative", PrecipitativeHyperparameters)
def train_precipitative_model(
    hyperparameters: PrecipitativeHyperparameters,
    train_batches: Sequence[xr.Dataset],
    validation_batches: Sequence[xr.Dataset],
):
    random_state = np.random.RandomState(np.random.get_state()[1][0])
    stacked_train_batches = StackedBatches(train_batches, random_state)
    stacked_validation_batches = StackedBatches(validation_batches, random_state)
    training_obj = PrecipitativeModel(hyperparameters=hyperparameters)
    training_obj.fit_statistics(stacked_train_batches[0])
    if len(stacked_validation_batches) > 0:
        val_batch = stacked_validation_batches[0]
    else:
        val_batch = None
    training_obj.fit(stacked_train_batches, validation_batch=val_batch)
    return training_obj.predictor


class PrecipitativeModel:
    def __init__(self, hyperparameters: PrecipitativeHyperparameters):
        input_variables = tuple(
            [T_NAME, Q_NAME, DELP_NAME, PHYS_PRECIP_NAME]
            + list(hyperparameters.additional_input_variables)
        )
        self._dense_network = hyperparameters.dense_network
        self._loss_type = hyperparameters.loss
        self.output_variables = (T_TENDENCY_NAME, Q_TENDENCY_NAME, PRECIP_NAME)
        self._couple_precip_to_dQ1_dQ2 = hyperparameters.couple_precip_to_dQ1_dQ2
        self.sample_dim_name = SAMPLE_DIM_NAME
        self.input_variables = input_variables
        self.input_packer = ArrayPacker(self.sample_dim_name, input_variables)
        self.humidity_packer = ArrayPacker(self.sample_dim_name, [Q_TENDENCY_NAME])
        self.output_packer = ArrayPacker(self.sample_dim_name, self.output_variables)
        output_without_precip = (T_TENDENCY_NAME, Q_TENDENCY_NAME)
        self.output_without_precip_packer = ArrayPacker(
            self.sample_dim_name, output_without_precip
        )
        self.input_scaler = LayerStandardScaler()
        self.output_scaler = LayerStandardScaler()
        self.output_without_precip_scaler = LayerStandardScaler()
        self.humidity_scaler = LayerStandardScaler()
        self._train_model: Optional[tf.keras.Model] = None
        self._predict_model: Optional[tf.keras.Model] = None
        self._training_loop = hyperparameters.training_loop
        self._statistics_are_fit = False
        self._optimizer = hyperparameters.optimizer_config.instance
        self._residual_regularizer = (
            hyperparameters.residual_regularizer_config.instance
        )
        if hyperparameters.weights is None:
            self.weights: Mapping[str, Union[int, float, np.ndarray]] = {}

    def fit_statistics(self, X: xr.Dataset):
        """
        Given a dataset with [sample, z] and [sample] arrays, fit the
        scalers and packers.
        """
        inputs = self.input_packer.to_array(X)
        self.input_scaler.fit(inputs)
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

    def _build_model(self) -> Tuple[tf.keras.Model, tf.keras.Model]:
        """
        Construct model for training and model for prediction which share weights.

        When you train the training model, weights in the prediction model also
        get updated.
        """
        input_layers, input_vector = get_input_vector(
            self.input_packer, n_window=None, series=False
        )
        norm_input_vector = self.input_scaler.normalize_layer(input_vector)
        output_features = sum(self.output_without_precip_packer.feature_counts.values())
        dense_network = self._dense_network.build(norm_input_vector, output_features)
        regularized_output = tf.keras.layers.Dense(
            output_features,
            activation="linear",
            activity_regularizer=self._residual_regularizer,
            name=f"regularized_output",
        )(dense_network.hidden_outputs[-1])
        denormalized_output = self.output_without_precip_scaler.denormalize_layer(
            regularized_output
        )
        unpacked_output = get_unpack_layer(
            self.output_without_precip_packer, feature_dim=1
        )(denormalized_output)
        assert self.output_without_precip_packer.pack_names[0] == T_TENDENCY_NAME
        assert self.output_without_precip_packer.pack_names[1] == Q_TENDENCY_NAME
        T_tendency = unpacked_output[0]
        q_tendency = unpacked_output[1]

        # share hidden layers with the dense network,
        # but no activity regularization for column precipitation
        norm_column_precip_vector = tf.keras.layers.Dense(
            self.humidity_packer.feature_counts[Q_TENDENCY_NAME], activation="linear",
        )(dense_network.hidden_outputs[-1])
        column_precip = self.humidity_scaler.denormalize_layer(
            norm_column_precip_vector
        )
        if self._couple_precip_to_dQ1_dQ2:
            column_heating = tf.keras.layers.Lambda(condensational_heating)(
                column_precip
            )
            T_tendency = tf.keras.layers.Add(name="T_tendency")(
                [T_tendency, column_heating]
            )
            q_tendency = tf.keras.layers.Add(name="q_tendency")(
                [q_tendency, column_precip]
            )

        delp = input_layers[self.input_variables.index(DELP_NAME)]
        physics_precip = input_layers[self.input_variables.index(PHYS_PRECIP_NAME)]
        surface_precip = tf.keras.layers.Add(name="add_physics_precip")(
            [
                physics_precip,
                tf.keras.layers.Lambda(integrate_precip, name="surface_precip")(
                    [column_precip, delp]
                ),
            ]
        )
        # This assertion is here to remind you that if you change the output variables
        # (including their order), you have to update this function. Look for places
        # where physical quantities are grabbed as indexes of output lists
        # (e.g. T_tendency = unpacked_output[0]).
        assert list(self.output_variables) == [
            T_TENDENCY_NAME,
            Q_TENDENCY_NAME,
            PRECIP_NAME,
        ], self.output_variables
        train_model = tf.keras.Model(
            inputs=input_layers, outputs=(T_tendency, q_tendency, surface_precip)
        )
        train_model.compile(
            optimizer=self._optimizer,
            loss=get_losses(
                self.output_variables,
                self.output_packer,
                self.output_scaler,
                loss_type=self._loss_type,
                weights=self.weights,
            ),
        )
        # need a separate model for this so we don't have to
        # serialize the custom loss functions
        predict_model = tf.keras.Model(
            inputs=input_layers, outputs=(T_tendency, q_tendency, surface_precip)
        )
        return train_model, predict_model

    def fit(
        self, batches: Sequence[xr.Dataset], validation_batch: Optional[xr.Dataset]
    ) -> None:
        """Fits a model using data in the batches sequence.
        """
        if not self._statistics_are_fit:
            raise RuntimeError("fit_statistics must be called before fit")
        if self._train_model is None:
            self._train_model, self._predict_model = self._build_model()
        if validation_batch is None:
            validation_data = None
        else:
            validation_data = (
                tuple(validation_batch[name].values for name in self.input_variables),
                tuple(validation_batch[name].values for name in self.output_variables),
            )
        Xy = _XyMultiArraySequence(self.input_variables, self.output_variables, batches)
        return self._training_loop.fit_loop(self._train_model, Xy, validation_data)

    @property
    def predictor(self) -> PureKerasModel:
        return PureKerasModel(
            self.sample_dim_name,
            self.input_variables,
            # predictor has additional diagnostic outputs which were indirectly trained
            list(self.output_variables),
            get_output_metadata(self.output_packer, self.sample_dim_name),
            self._predict_model,
        )
