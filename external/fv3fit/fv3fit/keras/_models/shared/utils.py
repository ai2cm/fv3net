from fv3fit._shared.packer import ArrayPacker
import tensorflow as tf
from typing import Optional


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
