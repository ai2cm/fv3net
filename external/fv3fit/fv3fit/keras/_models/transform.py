import dataclasses
from fv3fit._shared.predictor import Predictor
from toolz.functoolz import curry
import numpy as np
import tensorflow as tf
from typing import (
    Hashable,
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
    SliceConfig,
    register_training_function,
)
from .shared import (
    TrainingLoopConfig,
    DenseNetworkConfig,
)
from .dense import train_column_model
from fv3fit.keras._models.shared import (
    PureKerasModel,
    LossConfig,
    OutputLimitConfig,
    CallbackConfig,
)
from fv3fit.keras._models.shared.utils import (
    standard_denormalize,
    full_standard_normalized_input,
)
from fv3fit import tfdataset
from fv3fit.keras._models.shared.clip import clip_sequence, ClipConfig
from fv3fit.tfdataset import select_keys, ensure_nd, apply_to_mapping, clip_sample
from fv3fit.train_microphysics import TransformedParameters



@dataclasses.dataclass
class TransformNetworkConfig:
    """
    Attributes:
        width: number of neurons in each hidden layer
        depth: number of hidden layers + 1 (the output layer is a layer)
        kernel_regularizer: configuration of regularization for hidden layer weights
        gaussian_noise: amount of gaussian noise to add to each hidden layer output
        spectral_normalization: if True, apply spectral normalization to hidden layers
    """

    width: int = 8
    depth: int = 3
    kernel_regularizer: RegularizerConfig = dataclasses.field(
        default_factory=lambda: RegularizerConfig("none")
    )
    gaussian_noise: float = 0.0
    spectral_normalization: bool = False

    def build(
        self, x_in: tf.Tensor, n_features_out: int, label: str = ""
    ):
        """
        Take an input tensor to a dense network and return the result of a dense
        network's prediction, as a tensor.

        Can be used within code that builds a larger neural network. This should
        take in and return normalized values.

        Args:
            x_in: input tensor whose last dimension is the feature dimension
            n_features_out: dimensionality of last (feature) dimension of output
            config: configuration for dense network
            label: inserted into layer names, if this function is used multiple times
                to build one network you must provide a different label each time

        Returns:
            tensor resulting from the requested dense network
        """
        hidden_outputs = []
        x = x_in
        for i in range(self.depth - 1):
            if self.gaussian_noise > 0.0:
                x = tf.keras.layers.GaussianNoise(
                    self.gaussian_noise, name=f"gaussian_noise_{label}_{i}"
                )(x)
            hidden_layer = tf.keras.layers.Dense(
                self.width,
                activation=tf.keras.activations.relu,
                kernel_regularizer=self.kernel_regularizer.instance,
                name=f"hidden_{label}_{i}",
            )
            if self.spectral_normalization:
                hidden_layer = SpectralNormalization(
                    hidden_layer, name=f"spectral_norm_{label}_{i}"
                )
            x = hidden_layer(x)
            hidden_outputs.append(x)
        output = tf.keras.layers.Dense(
            n_features_out, activation="linear", name=f"dense_network_{label}_output",
        )(x)
        return DenseNetwork(hidden_outputs=hidden_outputs, output=output)


@dataclasses.dataclass
class TransformHyperparameters(Hyperparameters):
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
        clip_config: configuration of input and output clipping of last dimension
        output_limit_config: configuration for limiting output values.
        normalization_fit_samples: number of samples to use when fitting normalization
        callback_config: configuration for keras callbacks
    """

    input_variables: List[str]
    output_variables: List[str]
    weights: Optional[Mapping[str, Union[int, float]]] = None
    normalize_loss: bool = True
    optimizer_config: OptimizerConfig = dataclasses.field(
        default_factory=lambda: OptimizerConfig("Adam")
    )
    transform_network: TransformedParameters = dataclasses.field(
        default_factory=TransformedParameters
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
    normalization_fit_samples: int = 500_000
    callbacks: List[CallbackConfig] = dataclasses.field(default_factory=list)

    @property
    def variables(self) -> Set[str]:
        return set(self.input_variables).union(self.output_variables)


@register_training_function("build_transform", TransformHyperparameters)
def train_transform_model(
    hyperparameters: TransformHyperparameters,
    train_batches: tf.data.Dataset,
    validation_batches: Optional[tf.data.Dataset],
) -> Predictor:
    return train_column_model(
        train_batches=train_batches,
        validation_batches=validation_batches,
        build_model=curry(build_model)(config=hyperparameters),
        input_variables=hyperparameters.input_variables,
        output_variables=hyperparameters.output_variables,
        clip_config=hyperparameters.clip_config,
        training_loop=hyperparameters.training_loop,
        build_samples=hyperparameters.normalization_fit_samples,
        callbacks=hyperparameters.callbacks,
    )


def build_model(
    config: TransformHyperparameters, X: Tuple[tf.Tensor, ...], y: Tuple[tf.Tensor, ...]
) -> Tuple[tf.keras.Model, tf.keras.Model]:
    """
    Args:
        config: configuration of convolutional training
        X: example input for keras fitting, used to determine shape and normalization
        y: example output for keras fitting, used to determine shape and normalization
    """
    train_set = {name: value for name, value in zip(X, config.input_variables)}
    transform = config.transform_network.build_transform(train_set)
    model = config.transform_network.build_model(train_set, transform)
    return model, model
