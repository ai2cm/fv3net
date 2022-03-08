import dataclasses
from fv3fit.keras._models.dense import train_column_model
from toolz.functoolz import curry
from typing import List, Optional, Sequence, Tuple, Set
from fv3fit._shared.config import (
    OptimizerConfig,
    RegularizerConfig,
    register_training_function,
)
from fv3fit._shared.scaler import StandardScaler
from fv3fit.keras._models.shared.clip import ClipConfig, clip_sequence
from fv3fit.keras._models.shared.loss import LossConfig
import tensorflow as tf
from ..._shared import ArrayPacker
from ..._shared.config import Hyperparameters
from .shared import DenseNetworkConfig, TrainingLoopConfig
import numpy as np
import logging
from fv3fit.keras._models.shared.utils import (
    full_standard_normalized_input,
    standard_denormalize,
)


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


class IntegratePrecipLayer(tf.keras.layers.Layer):
    def call(self, args) -> tf.Tensor:
        """
        Args:
            dQ2: rate of change in moisture in kg/kg/s, negative corresponds
                to condensation
            delp: pressure thickness of atmospheric layer in Pa

        Returns:
            precipitation rate in kg/m^2/s
        """
        dQ2, delp = args  # layers seem to take in a single argument
        # output should be kg/m^2/s
        return tf.math.scalar_mul(
            # dQ2 * delp = s-1 * Pa = s^-1 * kg m-1 s-2 = kg m-1 s-3
            # dQ2 * delp / g = kg m-1 s-3 / (m s-2) = kg/m^2/s
            tf.constant(-1.0 / GRAVITY, dtype=dQ2.dtype),
            tf.math.reduce_sum(tf.math.multiply(dQ2, delp), axis=-1),
        )


class CondensationalHeatingLayer(tf.keras.layers.Layer):
    def call(self, dQ2: tf.Tensor) -> tf.Tensor:
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
        if loss_type == "mse":
            factor = tf.constant(
                1.0 / n_outputs / np.mean(std[name].values ** 2), dtype=tf.float32
            )
            loss = multiply_loss_by_factor(tf.losses.mse, factor)
        elif loss_type == "mae":
            factor = tf.constant(
                1.0 / n_outputs / np.mean(std[name].values), dtype=tf.float32
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
        clip_config: configuration of input clipping.
        loss: configuration of loss functions, will be applied separately to
            each output variable.
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
    clip_config: ClipConfig = dataclasses.field(default_factory=lambda: ClipConfig())
    loss: LossConfig = LossConfig(scaling="standard", loss_type="mse")
    couple_precip_to_dQ1_dQ2: bool = True

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

    @property
    def input_variables(self) -> Sequence[str]:
        return tuple(
            [T_NAME, Q_NAME, DELP_NAME, PHYS_PRECIP_NAME]
            + list(self.additional_input_variables)
        )

    @property
    def output_variables(self) -> Sequence[str]:
        return (T_TENDENCY_NAME, Q_TENDENCY_NAME, PRECIP_NAME)


@register_training_function("precipitative", PrecipitativeHyperparameters)
def train_precipitative_model(
    hyperparameters: PrecipitativeHyperparameters,
    train_batches: tf.data.Dataset,
    validation_batches: Optional[tf.data.Dataset],
):
    return train_column_model(
        train_batches=train_batches,
        validation_batches=validation_batches,
        build_model=curry(build_model)(config=hyperparameters),
        input_variables=hyperparameters.input_variables,
        output_variables=hyperparameters.output_variables,
        clip_config=hyperparameters.clip_config,
        training_loop=hyperparameters.training_loop,
    )


def build_model(
    config: PrecipitativeHyperparameters,
    X: Tuple[tf.Tensor, ...],
    y: Tuple[tf.Tensor, ...],
) -> Tuple[tf.keras.Model, tf.keras.Model]:
    """
    Construct model for training and model for prediction which share weights.

    When you train the training model, weights in the prediction model also
    get updated.
    """
    input_layers = [tf.keras.layers.Input(shape=arr.shape[1]) for arr in X]
    clipped_input_layers = clip_sequence(
        config.clip_config, input_layers, config.input_variables
    )
    # TODO: fold full_standard_normalized_input up to standard_denormalize
    # into a build method on config.dense_network, this process is duplicated
    # in three training routines
    full_input = full_standard_normalized_input(
        clipped_input_layers,
        clip_sequence(config.clip_config, X, config.input_variables),
        config.input_variables,
    )

    # note that n_features_out=1 is not used, as we want a separate output
    # layer for each variable (norm_output_layers)
    hidden_output = config.dense_network.build(
        full_input, n_features_out=1
    ).hidden_outputs[-1]

    norm_output_layers = [
        tf.keras.layers.Dense(
            array.shape[-1],
            activation="linear",
            activity_regularizer=config.residual_regularizer_config.instance,
            name=f"dense_network_output_{i}",
        )(hidden_output)
        for i, array in enumerate(y)
    ]
    unpacked_output = standard_denormalize(
        names=config.output_variables, layers=norm_output_layers, arrays=y,
    )
    assert config.output_variables[0] == T_TENDENCY_NAME
    assert config.output_variables[1] == Q_TENDENCY_NAME
    T_tendency = unpacked_output[0]
    q_tendency = unpacked_output[1]

    # share hidden layers with the dense network,
    # but no activity regularization for column precipitation
    i_q_tendency = config.output_variables.index(Q_TENDENCY_NAME)
    norm_column_precip_vector = tf.keras.layers.Dense(
        y[i_q_tendency].shape[-1], activation="linear",
    )(hidden_output)
    column_precip = standard_denormalize(
        names=["q_tendency_due_to_precip"],
        layers=[norm_column_precip_vector],
        arrays=[y[i_q_tendency]],
    )[0]
    if config.couple_precip_to_dQ1_dQ2:
        column_heating = CondensationalHeatingLayer()(column_precip)
        T_tendency = tf.keras.layers.Add(name="T_tendency")(
            [T_tendency, column_heating]
        )
        q_tendency = tf.keras.layers.Add(name="q_tendency")([q_tendency, column_precip])

    delp = input_layers[config.input_variables.index(DELP_NAME)]
    physics_precip = input_layers[config.input_variables.index(PHYS_PRECIP_NAME)]
    surface_precip = tf.keras.layers.Add(name="add_physics_precip")(
        [
            physics_precip,
            IntegratePrecipLayer(name="surface_precip")([column_precip, delp]),
        ]
    )
    # This assertion is here to remind you that if you change the output variables
    # (including their order), you have to update this function. Look for places
    # where physical quantities are grabbed as indexes of output lists
    # (e.g. T_tendency = unpacked_output[0]).
    assert list(config.output_variables) == [
        T_TENDENCY_NAME,
        Q_TENDENCY_NAME,
        PRECIP_NAME,
    ], config.output_variables
    train_model = tf.keras.Model(
        inputs=input_layers, outputs=(T_tendency, q_tendency, surface_precip)
    )
    output_stds = (
        np.std(array, axis=tuple(range(len(array.shape) - 1)), dtype=np.float32)
        for array in y
    )
    train_model.compile(
        optimizer=config.optimizer_config.instance,
        loss=[config.loss.loss(std) for std in output_stds],
    )
    # need a separate model for this so we don't have to
    # serialize the custom loss functions
    predict_model = tf.keras.Model(
        inputs=input_layers, outputs=(T_tendency, q_tendency, surface_precip)
    )
    return train_model, predict_model
