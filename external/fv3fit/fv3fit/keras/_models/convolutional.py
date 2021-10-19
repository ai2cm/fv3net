import dataclasses
from typing import Any, Dict, List, Sequence, Tuple, Set
from fv3fit._shared.config import (
    OptimizerConfig,
    register_training_function,
)
from fv3fit.keras._models.shared.loss import LossConfig
import tensorflow as tf
import xarray as xr
from ..._shared.config import Hyperparameters
from ._sequences import _XyMultiArraySequence
from .shared import ConvolutionalNetworkConfig, TrainingLoopConfig
import numpy as np
from fv3fit.keras._models.shared import PureKerasNoStackModel
import logging
from fv3fit.emulation.layers import StandardNormLayer, StandardDenormLayer

logger = logging.getLogger(__file__)


def multiply_loss_by_factor(original_loss, factor):
    def loss(y_true, y_pred):
        return tf.math.scalar_mul(factor, original_loss(y_true, y_pred))

    return loss


def get_metadata(names, ds: xr.Dataset) -> Tuple[Dict[str, Any]]:
    """
    Retrieve xarray metadata.

    Returns a dict containing "dims" and "units" for each name.
    """
    metadata = []
    for name in names:
        metadata.append(
            {"dims": ds[name].dims, "units": ds[name].attrs.get("units", "unknown")}
        )
    return tuple(metadata)


@dataclasses.dataclass
class ConvolutionalHyperparameters(Hyperparameters):
    """
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
        loss: loss function to use, should be 'mse' or 'mae'
        save_model_checkpoints: if True, save one model per epoch when
            dumping, under a 'model_checkpoints' subdirectory
        nonnegative_outputs: if True, add a ReLU activation layer as the last layer
            after output denormalization layer to ensure outputs are always >=0
            Defaults to False.
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
    loss: LossConfig = LossConfig(scaling="standard")
    couple_precip_to_dQ1_dQ2: bool = True

    @property
    def variables(self) -> Set[str]:
        return set(self.input_variables).union(self.output_variables)


@register_training_function("convolutional", ConvolutionalHyperparameters)
def train_convolutional_model(
    hyperparameters: ConvolutionalHyperparameters,
    train_batches: Sequence[xr.Dataset],
    validation_batches: Sequence[xr.Dataset],
):
    validation_data = batch_to_array_tuple(
        validation_batches[0],
        input_variables=hyperparameters.input_variables,
        output_variables=hyperparameters.output_variables,
    )
    train_data = _XyMultiArraySequence(
        X_names=hyperparameters.input_variables,
        y_names=hyperparameters.output_variables,
        dataset_sequence=train_batches,
    )
    train_model, predict_model = build_model(hyperparameters, batch=train_batches[0])
    hyperparameters.training_loop.fit_loop(
        model=train_model, Xy=train_data, validation_data=validation_data
    )
    output_metadata = get_metadata(
        names=hyperparameters.output_variables, ds=validation_batches[0]
    )
    predictor = PureKerasNoStackModel(
        input_variables=hyperparameters.input_variables,
        output_variables=hyperparameters.output_variables,
        output_metadata=output_metadata,
        model=predict_model,
    )
    return predictor


def batch_to_array_tuple(
    batch: xr.Dataset, input_variables: Sequence[str], output_variables: Sequence[str]
) -> Tuple[Tuple[np.ndarray, ...], Tuple[np.ndarray, ...]]:
    return (
        tuple(batch[name].values for name in input_variables),
        tuple(batch[name].values for name in output_variables),
    )


def count_features(names, batch: xr.Dataset):
    """
    Retrieves a function that takes in a packed tensor and returns a sequence of
    unpacked tensors, having been split by keras Layers.

    Args:
        names: dataset keys to be unpacked
        batch: dataset containing representatively-shaped data for the given names,
            last dimension should be the feature dimension.
    """
    feature_counts = []
    for name in names:
        feature_counts.append(batch[name].shape[-1])
    return feature_counts


def standard_normalize(names, layers, batch: xr.Dataset):
    out = []
    for name, layer in zip(names, layers):
        norm = StandardNormLayer(name=f"standard_normalize_{name}")
        norm.fit(batch[name].values)
        out.append(norm(layer))
    return out


def standard_denormalize(names, layers, batch: xr.Dataset):
    out = []
    for name, layer in zip(names, layers):
        norm = StandardDenormLayer(name=f"standard_denormalize_{name}")
        norm.fit(batch[name].values)
        out.append(norm(layer))
    return out


def build_model(
    config: ConvolutionalHyperparameters, batch: xr.Dataset
) -> Tuple[tf.keras.Model, tf.keras.Model]:
    nx = batch.dims["x"]
    ny = batch.dims["y"]
    input_features = count_features(config.input_variables, batch)
    input_layers = [
        tf.keras.layers.Input(shape=(nx, ny, n_feature)) for n_feature in input_features
    ]
    norm_input_layers = standard_normalize(
        names=config.input_variables, layers=input_layers, batch=batch
    )
    if len(norm_input_layers) > 1:
        full_input = tf.keras.layers.Concatenate()(norm_input_layers)
    else:
        full_input = norm_input_layers[0]
    convolution = config.convolutional_network.build(x_in=full_input, n_features_out=0)
    output_features = count_features(config.output_variables, batch)
    norm_output_layers = (
        tf.keras.layers.Conv2D(
            filters=n_features,
            kernel_size=(1, 1),
            padding="same",
            activation="linear",
            data_format="channels_last",
            name=f"convolutional_network_{i}_output",
        )(convolution.hidden_outputs[-1])
        for i, n_features in enumerate(output_features)
    )
    denorm_output_layers = standard_denormalize(
        names=config.output_variables, layers=norm_output_layers, batch=batch
    )
    train_model = tf.keras.Model(inputs=input_layers, outputs=denorm_output_layers)
    output_stds = (
        np.std(
            batch[name], axis=tuple(range(len(batch[name].shape) - 1)), dtype=np.float32
        )
        for name in config.output_variables
    )
    train_model.compile(
        optimizer=config.optimizer_config.instance,
        loss=[config.loss.loss(std) for std in output_stds],
    )
    # need a separate model for this so we don't have to
    # serialize the custom loss functions
    predict_model = tf.keras.Model(inputs=input_layers, outputs=denorm_output_layers)
    return train_model, predict_model
