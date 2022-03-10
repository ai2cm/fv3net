from fv3fit._shared.packer import ArrayPacker
import tensorflow as tf
from typing import List, Optional, Sequence, Type
from fv3fit.emulation.layers import StandardNormLayer, StandardDenormLayer, NormLayer

import numpy as np


def get_input_vector(
    packer: ArrayPacker, n_window: Optional[int] = None, series: bool = True,
):
    """
    Given a packer, return a list of input layers with one layer
    for each input used by the packer, and a list of output tensors which are
    the result of packing those input layers.

    Args:
        packer
        n_window: required if series is True, number of timesteps in a sample
        series: if True, returned inputs have shape [n_window, n_features], otherwise
            they are 1D [n_features] arrays
    """
    features = [packer.feature_counts[name] for name in packer.pack_names]
    if series:
        if n_window is None:
            raise TypeError("n_window is required if series is True")
        input_layers = [
            tf.keras.layers.Input(shape=[n_window, n_features])
            for n_features in features
        ]
    else:
        input_layers = [
            tf.keras.layers.Input(shape=[n_features]) for n_features in features
        ]
    packed = tf.keras.layers.Concatenate()(input_layers)
    return input_layers, packed


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
