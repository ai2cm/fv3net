import dataclasses
from typing import List, Optional, Sequence, Tuple, Set
from fv3fit._shared.config import (
    OptimizerConfig,
    register_training_function,
)
from fv3fit.keras._models.shared.loss import LossConfig
from fv3fit.keras._models.shared.pure_keras import PureKerasModel
import tensorflow as tf
import xarray as xr
from ..._shared.config import Hyperparameters
from .shared import ConvolutionalNetworkConfig, TrainingLoopConfig, XyMultiArraySequence
import numpy as np
from fv3fit.keras._models.shared import Diffusive
import logging
from fv3fit.keras._models.shared.utils import (
    standard_denormalize,
    full_standard_normalized_input,
)

logger = logging.getLogger(__file__)

UNSTACKED_DIMS = ("x", "y", "z", "z_interface")


def multiply_loss_by_factor(original_loss, factor):
    def loss(y_true, y_pred):
        return tf.math.scalar_mul(factor, original_loss(y_true, y_pred))

    return loss


@dataclasses.dataclass
class ConvolutionalHyperparameters(Hyperparameters):
    """
    Args:
        input_variables: names of variables to use as inputs
        output_variables: names of variables to use as outputs
        optimizer_config: selection of algorithm to be used in gradient descent
        convolutional_network: configuration of convolutional network
        training_loop: configuration of training loop
        loss: configuration of loss functions, will be applied separately to
            each output variable
    """

    input_variables: List[str]
    output_variables: List[str]
    optimizer_config: OptimizerConfig = dataclasses.field(
        default_factory=lambda: OptimizerConfig("Adam")
    )
    convolutional_network: ConvolutionalNetworkConfig = dataclasses.field(
        default_factory=lambda: ConvolutionalNetworkConfig()
    )
    training_loop: TrainingLoopConfig = dataclasses.field(
        default_factory=lambda: TrainingLoopConfig(epochs=10)
    )
    loss: LossConfig = LossConfig(scaling="standard", loss_type="mse")

    @property
    def variables(self) -> Set[str]:
        return set(self.input_variables).union(self.output_variables)


@register_training_function("convolutional", ConvolutionalHyperparameters)
def train_convolutional_model(
    hyperparameters: ConvolutionalHyperparameters,
    train_batches: Sequence[xr.Dataset],
    validation_batches: Optional[Sequence[xr.Dataset]] = None,
):
    n_halo = hyperparameters.convolutional_network.halos_required
    if validation_batches is not None and len(validation_batches) > 0:
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = XyMultiArraySequence(
            X_names=hyperparameters.input_variables,
            y_names=hyperparameters.output_variables,
            dataset_sequence=[validation_batches[0]],
            unstacked_dims=UNSTACKED_DIMS,
            n_halo=n_halo,
        )[0]
        del validation_batches
    else:
        validation_data = None
    train_data: Sequence[Tuple[np.ndarray, np.ndarray]] = XyMultiArraySequence(
        X_names=hyperparameters.input_variables,
        y_names=hyperparameters.output_variables,
        dataset_sequence=train_batches,
        unstacked_dims=UNSTACKED_DIMS,
        n_halo=n_halo,
    )
    if isinstance(train_batches, tuple):
        train_data = tuple(train_data)
    X, y = train_data[0]
    train_model, predict_model = build_model(hyperparameters, X=X, y=y)
    del train_batches
    hyperparameters.training_loop.fit_loop(
        model=train_model, Xy=train_data, validation_data=validation_data
    )
    predictor = PureKerasModel(
        input_variables=hyperparameters.input_variables,
        output_variables=hyperparameters.output_variables,
        model=predict_model,
        unstacked_dims=("x", "y", "z"),
        n_halo=n_halo,
    )
    return predictor


def _ensure_4d(array: np.ndarray) -> np.ndarray:
    # Ensures [sample, x, y, z] dimensionality.
    if len(array.shape) == 4:
        return array
    elif len(array.shape) == 3:
        return array[:, :, :, None]
    else:
        raise ValueError(f"expected 3d or 4d array, got shape {array.shape}")


def build_model(
    config: ConvolutionalHyperparameters,
    X: Sequence[np.ndarray],
    y: Sequence[np.ndarray],
) -> Tuple[tf.keras.Model, tf.keras.Model]:
    """
    Args:
        config: configuration of convolutional training
        X: example input for keras fitting, used to determine shape and normalization
        y: example output for keras fitting, used to determine shape and normalization
    """
    X_4d = [_ensure_4d(arr) for arr in X]
    y_4d = [_ensure_4d(arr) for arr in y]

    input_layers = [tf.keras.layers.Input(shape=arr.shape[1:]) for arr in X_4d]
    full_input = full_standard_normalized_input(
        input_layers, X_4d, config.input_variables
    )

    convolution = config.convolutional_network.build(x_in=full_input, n_features_out=0)
    if config.convolutional_network.diffusive:
        constraint: Optional[tf.keras.constraints.Constraint] = Diffusive()
    else:
        constraint = None
    norm_output_layers = [
        tf.keras.layers.Conv2D(
            filters=array.shape[-1],
            kernel_size=(1, 1),
            padding="same",
            activation="linear",
            data_format="channels_last",
            name=f"convolutional_network_{i}_output",
            kernel_constraint=constraint,
        )(convolution.hidden_outputs[-1])
        for i, array in enumerate(y_4d)
    ]
    denorm_output_layers = standard_denormalize(
        names=config.output_variables, layers=norm_output_layers, arrays=y_4d,
    )
    train_model = tf.keras.Model(inputs=input_layers, outputs=denorm_output_layers)
    output_stds = (
        np.std(array, axis=tuple(range(len(array.shape) - 1)), dtype=np.float32)
        for array in y_4d
    )
    train_model.compile(
        optimizer=config.optimizer_config.instance,
        loss=[config.loss.loss(std) for std in output_stds],
    )
    # need a separate model for this so we don't have to
    # serialize the custom loss functions
    predict_model = tf.keras.Model(inputs=input_layers, outputs=denorm_output_layers)
    return train_model, predict_model
