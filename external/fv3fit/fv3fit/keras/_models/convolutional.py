import dataclasses
from typing import Hashable, List, Mapping, Optional, Sequence, Tuple, Set
from fv3fit._shared.config import (
    OptimizerConfig,
    register_training_function,
)
from fv3fit.keras._models.shared.halos import append_halos_tensor
from toolz.functoolz import curry
from fv3fit._shared.packer import clip_sample
from fv3fit.keras._models.shared.loss import LossConfig
from fv3fit.keras._models.shared.pure_keras import PureKerasModel
import tensorflow as tf
from ..._shared.config import Hyperparameters, SliceConfig
from .shared import ConvolutionalNetworkConfig, TrainingLoopConfig
import numpy as np
from fv3fit.keras._models.shared import Diffusive
import logging
from fv3fit.keras._models.shared.utils import (
    standard_denormalize,
    full_standard_normalized_input,
)
from fv3fit.tfdataset import select_keys, ensure_nd, apply_to_mapping

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
        default_factory=lambda: TrainingLoopConfig(epochs=10, batch_size=1)
    )
    clip_config: Mapping[Hashable, SliceConfig] = dataclasses.field(
        default_factory=dict
    )
    loss: LossConfig = LossConfig(scaling="standard", loss_type="mse")

    @property
    def variables(self) -> Set[str]:
        return set(self.input_variables).union(self.output_variables)


def get_Xy_dataset(
    input_variables: Sequence[str],
    output_variables: Sequence[str],
    clip_config: Optional[Mapping[Hashable, SliceConfig]],
    n_halo: int,
    data: tf.data.Dataset,
):
    """
    Given a tf.data.Dataset with mappings from variable name to samples,
    return a tf.data.Dataset whose entries are two tuples, the first containing the
    requested input variables and the second containing
    the requested output variables.
    """
    # tile, x, y, z
    data = data.map(apply_to_mapping(ensure_nd(4)))
    if clip_config is not None:
        y_source = data.map(clip_sample(clip_config))
    else:
        y_source = data
    y = y_source.map(select_keys(output_variables))
    X = data.map(apply_to_mapping(append_halos_tensor(n_halo))).map(
        select_keys(input_variables)
    )
    # now that we have halos, we need to collapse tile back into the sample dimension
    X = X.unbatch()
    y = y.unbatch()
    return tf.data.Dataset.zip((X, y))


@register_training_function("convolutional", ConvolutionalHyperparameters)
def train_convolutional_model(
    hyperparameters: ConvolutionalHyperparameters,
    train_batches: tf.data.Dataset,
    validation_batches: Optional[tf.data.Dataset] = None,
):
    get_Xy = curry(get_Xy_dataset)(
        input_variables=hyperparameters.input_variables,
        output_variables=hyperparameters.output_variables,
        n_halo=hyperparameters.convolutional_network.halos_required,
    )
    if validation_batches is not None:
        val_Xy = get_Xy(
            clip_config=hyperparameters.clip_config, data=validation_batches
        )
    else:
        val_Xy = None

    train_Xy = get_Xy(data=train_batches, clip_config=hyperparameters.clip_config)
    # need unclipped shapes for build_model
    X, y = next(iter(get_Xy(data=train_batches, clip_config=None).batch(10_000_000)))

    train_model, predict_model = build_model(hyperparameters, X=X, y=y)
    hyperparameters.training_loop.fit_loop(
        model=train_model, Xy=train_Xy, validation_data=val_Xy
    )
    predictor = PureKerasModel(
        input_variables=hyperparameters.input_variables,
        output_variables=hyperparameters.output_variables,
        model=predict_model,
        # we train with a tile dim, but in the prognostic run each rank
        # only has local data and never has a tile dimension
        unstacked_dims=("x", "y", "z"),
        n_halo=hyperparameters.convolutional_network.halos_required,
    )
    return predictor


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

    input_layers = [tf.keras.layers.Input(shape=arr.shape[1:]) for arr in X]
    full_input = full_standard_normalized_input(input_layers, X, config.input_variables)
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
        for i, array in enumerate(y)
    ]
    denorm_output_layers = standard_denormalize(
        names=config.output_variables, layers=norm_output_layers, arrays=y,
    )
    train_model = tf.keras.Model(inputs=input_layers, outputs=denorm_output_layers)
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
    predict_model = tf.keras.Model(inputs=input_layers, outputs=denorm_output_layers)
    return train_model, predict_model
