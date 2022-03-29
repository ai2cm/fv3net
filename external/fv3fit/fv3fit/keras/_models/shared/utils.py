import tensorflow as tf
from typing import List, Sequence, Type
from fv3fit.emulation.layers import StandardNormLayer, StandardDenormLayer, NormLayer

import numpy as np


def _fit_norm_layer(
    cls: Type[NormLayer],
    names: Sequence[str],
    layers: Sequence[tf.Tensor],
    arrays: Sequence[np.ndarray],
) -> Sequence[NormLayer]:
    out: List[NormLayer] = []
    for name, layer, array in zip(names, layers, arrays):
        norm = cls(name=f"standard_normalize_{name}")
        norm.fit(array)
        out.append(norm(layer))
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
    return _fit_norm_layer(StandardNormLayer, names=names, layers=layers, arrays=arrays)


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
    return _fit_norm_layer(
        StandardDenormLayer,  # type: ignore
        names=names,
        layers=layers,
        arrays=arrays,
    )


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
