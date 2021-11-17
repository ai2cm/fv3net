import dataclasses
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from typing import List, Optional, Sequence, Tuple, Set, Mapping, Union
import xarray as xr

from ..._shared.config import (
    Hyperparameters,
    OptimizerConfig,
    register_training_function,
    PackerConfig
)
from .shared import TrainingLoopConfig, XyMultiArraySequence, DenseNetworkConfig
from fv3fit.keras._models.shared import PureKerasModel, LossConfig
from fv3fit.keras._models.shared.utils import standard_denormalize, standard_normalize, get_stacked_metadata



@dataclasses.dataclass
class DenseHyperparameters(Hyperparameters):
    """
    Configuration for training a dense neural network based model.

    Args:
        input_variables: names of variables to use as inputs
        output_variables: names of variables to use as outputs
        weights: loss function weights, defined as a dict whose keys are
            variable names and values are either a scalar referring to the total
            weight of the variable. Default is a total weight of 1
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
        nonnegative_outputs: if True, add a ReLU activation layer as the last layer
            after output denormalization layer to ensure outputs are always >=0
            Defaults to False.
        packer_config: configuration of dataset packing.
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
    nonnegative_outputs: bool = False
    packer_config: PackerConfig = dataclasses.field(
        default_factory=lambda: PackerConfig({})
    )

    @property
    def variables(self) -> Set[str]:
        return set(self.input_variables).union(self.output_variables)


@register_training_function("dense", DenseHyperparameters)
def train_dense_model(
    hyperparameters: DenseHyperparameters,
    train_batches: Sequence[xr.Dataset],
    validation_batches: Optional[Sequence[xr.Dataset]] = None,
):
    if validation_batches is not None and len(validation_batches) > 0:
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = XyMultiArraySequence(
            X_names=hyperparameters.input_variables,
            y_names=hyperparameters.output_variables,
            dataset_sequence=[validation_batches[0]],
            unstacked_dims=["z",],
            n_halo=0,
        )[0]
        del validation_batches
    else:
        validation_data = None
    train_data: Sequence[Tuple[np.ndarray, np.ndarray]] = XyMultiArraySequence(
        X_names=hyperparameters.input_variables,
        y_names=hyperparameters.output_variables,
        dataset_sequence=train_batches,
        unstacked_dims=["z",],
        n_halo=0,
    )
    if isinstance(train_batches, tuple):
        train_data = tuple(train_data)
    X, y = train_data[0]

    train_model, predict_model = build_model(hyperparameters, X=X, y=y)

    output_metadata = get_stacked_metadata(
        names=hyperparameters.output_variables,
        ds=train_batches[0],
        unstacked_dims="z",
    )
    del train_batches
    hyperparameters.training_loop.fit_loop(
        model=train_model, Xy=train_data, validation_data=validation_data
    )
    predictor = PureKerasModel(
        input_variables=hyperparameters.input_variables,
        output_variables=hyperparameters.output_variables,
        output_metadata=output_metadata,
        model=predict_model,
        unstacked_dims=("z",),
        n_halo=0,
    )
    return predictor




def _count_array_features(arr: np.ndarray) -> int:
    """
    Given a 1/2-d [sample, (, z)] array, return its number of
    vertical levels.
    """
    if len(arr.shape) == 1:
        return 1
    elif len(arr.shape) == 2:
        return arr.shape[-1]
    else:
        raise ValueError(f"expected 1d or 4d array, got shape {arr.shape}")


def _ensure_2d(array: np.ndarray) -> np.ndarray:
    # ensures [sample, z] dimensionality
    if len(array.shape) == 2:
        return array
    elif len(array.shape) == 1:
        return array[:, None]
    else:
        raise ValueError(f"expected 1d or 2d array, got shape {array.shape}")


def _get_input_layer_shapes(X: Sequence[np.ndarray]) -> List[Tuple[int]]:
    # adds a z dim of length 1 for non vertical features, so that they can be concantenated
    # with inputs that have a z dimension
    shapes: List[Tuple[int]] = []
    for array in X:
        array_features = _count_array_features(array)
        sample_shape: Tuple[int] = array.shape[1:]
        if array_features == 1:
            shapes.append((*sample_shape, 1))  # type: ignore
        else:
            shapes.append(sample_shape)
    return shapes


def build_model(config: DenseHyperparameters, X, y) ->  Tuple[tf.keras.Model, tf.keras.Model]:
    """
    Args:
        config: configuration of convolutional training
        X: example input for keras fitting, used to determine shape and normalization
        y: example output for keras fitting, used to determine shape and normalization
    """
    input_layer_shapes = _get_input_layer_shapes(X)
    input_layers = [
        tf.keras.layers.Input(shape=input_shape) for input_shape in input_layer_shapes
    ]
    norm_input_layers = standard_normalize(
        names=config.input_variables,
        layers=input_layers,
        arrays=[_ensure_2d(array) for array in X],
    )
    if len(norm_input_layers) > 1:
        full_input = tf.keras.layers.Concatenate()(norm_input_layers)
    else:
        full_input = norm_input_layers[0]

    hidden_outputs = []

    x = full_input
    for i in range(config.dense_network.depth - 1):
        if config.dense_network.gaussian_noise > 0.0:
            x = tf.keras.layers.GaussianNoise(
                config.dense_network.gaussian_noise, name=f"gaussian_noise_{i}"
            )(x)
        hidden_layer = tf.keras.layers.Dense(
            config.dense_network.width,
            activation=tf.keras.activations.relu,
            kernel_regularizer=config.dense_network.kernel_regularizer.instance,
            name=f"hidden_{i}",
        )
        if config.dense_network.spectral_normalization:
            hidden_layer = tfa.layers.SpectralNormalization(
                hidden_layer, name=f"spectral_norm_{i}"
            )
        x = hidden_layer(x)
        hidden_outputs.append(x)

    norm_output_layers = [
        tf.keras.layers.Dense(
            _count_array_features(array), activation="linear", name=f"dense_network_output_{i}",
        )(hidden_outputs[-1])
        for i, array in enumerate(y)
    ]
    if config.nonnegative_outputs is True:
        norm_output_layers = [
            tf.keras.layers.Activation(tf.keras.activations.relu)(output_layer)
            for output_layer in norm_output_layers
        ]

    y_2d = [_ensure_2d(array) for array in y]
    denorm_output_layers = standard_denormalize(
        names=config.output_variables, layers=norm_output_layers, arrays=y_2d,
    )

    train_model = tf.keras.Model(inputs=input_layers, outputs=denorm_output_layers)
    output_stds = (
        np.std(array, axis=tuple(range(len(array.shape) - 1)), dtype=np.float32)
        for array in y_2d
    )
    train_model.compile(
        optimizer=config.optimizer_config.instance,
        loss=[config.loss.loss(std) for std in output_stds],
    )
    # need a separate model for this so we don't have to
    # serialize the custom loss functions
    predict_model = tf.keras.Model(inputs=input_layers, outputs=denorm_output_layers)
    return train_model, predict_model