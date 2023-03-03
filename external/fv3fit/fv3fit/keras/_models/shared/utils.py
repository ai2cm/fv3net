import tensorflow as tf
from typing import Sequence
from fv3fit.emulation.layers.normalization import (
    NormFactory,
    NormLayer,
    MeanMethod,
    StdDevMethod,
)
import numpy as np


def _build_norm_layers(
    factory: NormFactory, names: Sequence[str], arrays: Sequence[np.ndarray],
):
    norm_layers = []
    for name, array in zip(names, arrays):
        norm_layers.append(factory.build(array, name=name))
    return norm_layers


def _apply_norm_layers(
    norm_layers: Sequence[NormLayer], input_layers: Sequence[tf.Tensor], forward: bool,
):
    out = []
    for norm, in_ in zip(norm_layers, input_layers):
        if forward:
            tensor_func = norm.forward
        else:
            tensor_func = norm.backward
        out.append(tensor_func(in_))
    return out


def standard_normalize(
    names: Sequence[str], layers: Sequence[tf.Tensor], arrays: Sequence[np.ndarray],
) -> Sequence[tf.Tensor]:
    """
    Apply standard scaling to a series of layers based on mean and standard
    deviation from input arrays.

    Args:
        names: variable name in batch of each layer in layers
        layers: input tensors to be scaled by scaling layers
        arrays: arrays whose last dimension is feature dimension
            (possibly of length 1) on which to fit statistics

    Returns:
        normalized_layers: standard-scaled tensors
    """

    names = [f"standard_normalize_{name}" for name in names]
    factory = NormFactory(
        scale=StdDevMethod.per_feature, center=MeanMethod.per_feature, epsilon=1e-7
    )
    norm_layers = _build_norm_layers(factory, names, arrays)
    return _apply_norm_layers(norm_layers, layers, forward=True)


def standard_denormalize(
    names: Sequence[str], layers: Sequence[tf.Tensor], arrays: Sequence[np.ndarray],
) -> Sequence[tf.Tensor]:
    """
    Apply standard descaling to a series of standard-scaled
    layers based on mean and standard deviation from input arrays.

    Args:
        names: variable name in batch of each layer in layers
        layers: input tensors to be scaled by de-scaling layers
        arrays: arrays whose last dimension is feature dimension
            (possibly of length 1) on which to fit statistics

    Returns:
        denormalized_layers: de-scaled tensors
    """

    names = [f"standard_denormalize_{name}" for name in names]
    factory = NormFactory(
        scale=StdDevMethod.per_feature, center=MeanMethod.per_feature, epsilon=1e-7
    )
    norm_layers = _build_norm_layers(factory, names, arrays)
    return _apply_norm_layers(norm_layers, layers, forward=False)


def full_standard_normalized_input(
    input_layers: Sequence[tf.Tensor],
    X: Sequence[np.ndarray],
    input_variables: Sequence[str],
) -> tf.Tensor:
    # Takes in input arrays and returns a single input layer to standard
    # normalize and concatenate inputs together along the feature dimension.
    # All input arrays in X must have same rank, i.e. 2D quantities without
    # a feature dimension must have a last dim of size 1.
    input_ranks = [len(arr.shape) for arr in X]
    if len(np.unique(input_ranks)) > 1:
        raise ValueError(
            "All input arrays provided must have the same number of dimensions."
        )
    norm_input_layers = standard_normalize(
        names=input_variables, layers=input_layers, arrays=X,
    )
    if len(norm_input_layers) > 1:
        full_input = tf.keras.layers.Concatenate()(norm_input_layers)
    else:
        full_input = norm_input_layers[0]
    return full_input
