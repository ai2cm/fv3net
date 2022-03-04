import dataclasses
from toolz.functoolz import curry
import numpy as np
import tensorflow as tf
from typing import (
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Set,
    Mapping,
    Union,
)

from ..._shared.config import (
    Hyperparameters,
    OptimizerConfig,
    register_training_function,
)
from .shared import (
    TrainingLoopConfig,
    DenseNetworkConfig,
    TrainingLoopLossHistory,
)
from fv3fit.keras._models.shared import (
    PureKerasModel,
    LossConfig,
    OutputLimitConfig,
)
from fv3fit.keras._models.shared.utils import (
    standard_denormalize,
    full_standard_normalized_input,
)
from fv3fit.keras._models.shared.clip import clip_sequence, ClipConfig
from fv3fit.tfdataset import get_Xy_dataset


@dataclasses.dataclass
class DenseHyperparameters(Hyperparameters):
    """
    Configuration for training a dense neural network based model.

    Args:
        input_variables: names of variables to use as inputs
        output_variables: names of variables to use as outputs
        weights: loss function weights, defined as a dict whose keys are
            variable names and values are either a scalar referring to the
            total weight of the variable. Default is a total weight of 1
            for each variable.
        normalize_loss: if True (default), normalize outputs by their standard
            deviation before computing the loss function
        optimizer_config: selection of algorithm to be used in gradient descent
        dense_network: configuration of dense network
        training_loop: configuration of training loop
        loss: configuration of loss functions, will be applied separately to
            each output variable.
        save_model_checkpoints: if True, save one model per epoch when
            dumping, under a 'model_checkpoints' subdirectory
        clip_config: configuration of input and output clipping
        output_limit_config: configuration for limiting output values.
    """

    input_variables: List[str]
    output_variables: List[str]
    weights: Optional[Mapping[str, Union[int, float]]] = None
    normalize_loss: bool = True
    optimizer_config: OptimizerConfig = dataclasses.field(
        default_factory=lambda: OptimizerConfig("Adam")
    )
    dense_network: DenseNetworkConfig = dataclasses.field(
        default_factory=DenseNetworkConfig
    )
    training_loop: TrainingLoopConfig = dataclasses.field(
        default_factory=TrainingLoopConfig
    )
    loss: LossConfig = LossConfig(scaling="standard", loss_type="mse")
    save_model_checkpoints: bool = False
    clip_config: ClipConfig = dataclasses.field(default_factory=lambda: ClipConfig())
    output_limit_config: OutputLimitConfig = dataclasses.field(
        default_factory=lambda: OutputLimitConfig()
    )

    @property
    def variables(self) -> Set[str]:
        return set(self.input_variables).union(self.output_variables)


@register_training_function("dense", DenseHyperparameters)
def train_dense_model(
    hyperparameters: DenseHyperparameters,
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


class ModelBuilder(Protocol):
    def __call__(
        self, X: tf.Tensor, y: tf.Tensor
    ) -> Tuple[tf.keras.Model, tf.keras.Model]:
        ...


def train_column_model(
    train_batches: tf.data.Dataset,
    validation_batches: Optional[tf.data.Dataset],
    build_model: ModelBuilder,
    input_variables: Sequence[str],
    output_variables: Sequence[str],
    clip_config: ClipConfig,
    training_loop: TrainingLoopConfig,
):
    get_Xy = curry(get_Xy_dataset)(
        input_variables=input_variables, output_variables=output_variables, n_dims=1,
    )
    if validation_batches is not None:
        val_Xy = get_Xy(clip_config=clip_config.clip, data=validation_batches)
    else:
        val_Xy = None

    train_Xy = get_Xy(data=train_batches, clip_config=clip_config.clip)
    # need unclipped shapes for build_model
    X, y = next(iter(get_Xy(data=train_batches, clip_config=None).batch(10_000_000)))

    train_model, predict_model = build_model(X=X, y=y)

    loss_history = TrainingLoopLossHistory()
    training_loop.fit_loop(
        model=train_model,
        Xy=train_Xy,
        validation_data=val_Xy,
        callbacks=[loss_history.callback],
    )
    predictor = PureKerasModel(
        input_variables=input_variables,
        output_variables=output_variables,
        model=predict_model,
        unstacked_dims=("z",),
        n_halo=0,
    )
    loss_history.log_summary()
    return predictor


def build_model(
    config: DenseHyperparameters, X: Tuple[tf.Tensor, ...], y: Tuple[tf.Tensor, ...]
) -> Tuple[tf.keras.Model, tf.keras.Model]:
    """
    Args:
        config: configuration of convolutional training
        X: example input for keras fitting, used to determine shape and normalization
        y: example output for keras fitting, used to determine shape and normalization
    """
    input_layers = [tf.keras.layers.Input(shape=arr.shape[1]) for arr in X]
    clipped_input_layers = clip_sequence(
        config.clip_config, input_layers, config.input_variables
    )
    full_input = full_standard_normalized_input(
        clipped_input_layers,
        clip_sequence(config.clip_config, X, config.input_variables),
        config.input_variables,
    )

    # note that n_features_out=1 is not used, as we want a separate output
    # layer for each variable (norm_output_layers)
    hidden_outputs = config.dense_network.build(
        full_input, n_features_out=1
    ).hidden_outputs

    norm_output_layers = [
        tf.keras.layers.Dense(
            array.shape[-1], activation="linear", name=f"dense_network_output_{i}",
        )(hidden_outputs[-1])
        for i, array in enumerate(y)
    ]

    denorm_output_layers = standard_denormalize(
        names=config.output_variables, layers=norm_output_layers, arrays=y,
    )

    # Apply output range limiters
    denorm_output_layers = config.output_limit_config.apply_output_limiters(
        outputs=denorm_output_layers, names=config.output_variables
    )

    # Model used in training has output levels clipped off, so std also must
    # be calculated over the same set of levels after clipping.
    clipped_denorm_output_layers = clip_sequence(
        config.clip_config, denorm_output_layers, config.output_variables
    )
    clipped_output_arrays = clip_sequence(
        config.clip_config, list(y), config.output_variables
    )
    clipped_output_stds = (
        np.std(array, axis=tuple(range(len(array.shape) - 1)), dtype=np.float32)
        for array in clipped_output_arrays
    )

    train_model = tf.keras.Model(
        inputs=input_layers, outputs=clipped_denorm_output_layers
    )
    train_model.compile(
        optimizer=config.optimizer_config.instance,
        loss=[config.loss.loss(std) for std in clipped_output_stds],
    )

    # Returns a separate prediction model where outputs layers have
    # original length along last dimension, but with clipped levels masked to zero
    zero_filled_denorm_output_layers = [
        config.clip_config.zero_mask_clipped_layer(denorm_layer, name)
        for denorm_layer, name in zip(denorm_output_layers, config.output_variables)
    ]
    predict_model = tf.keras.Model(
        inputs=input_layers, outputs=zero_filled_denorm_output_layers
    )
    return train_model, predict_model
