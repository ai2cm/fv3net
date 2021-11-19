import dataclasses
import numpy as np
import tensorflow as tf
from typing import Mapping, Hashable, Sequence
from  fv3fit._shared.config import SliceConfig


ClipDims = Mapping[Hashable, Mapping[str, SliceConfig]]


@dataclasses.dataclass(frozen=True)
class ClipConfig:
    clip: ClipDims = dataclasses.field(default_factory=dict)

    def single_feature_slice(self, name: str) -> SliceConfig:
        self._check_only_feature_dim_is_clipped(name)   
        variable_slice_config = list(self.clip[name].values())[0]
        return variable_slice_config

    def removed_sizes(self, original_1d_layer_size: int, name: str):
        # Returns the size of the feature dim slices at the start and end of a
        # 1D layer that are REMOVED by clipping. Used to pad output layer back to their
        # original size.
        variable_slice_config = list(self.clip[name].values())[0]
        start = variable_slice_config.start or 0
        stop = variable_slice_config.stop or original_1d_layer_size
        n_removed_from_start = start
        n_removed_from_end = original_1d_layer_size - stop
        return (n_removed_from_start, n_removed_from_end) 

    def clip_layer_along_feature_dim(self, layer, name):
        if name in self.clip:
            self._check_only_feature_dim_is_clipped(name)   
            variable_slice_config = list(self.clip[name].values())[0]
            if variable_slice_config.step is not None:
                raise NotImplementedError(
                    "Stepped slice is not implented for clipping a 1D layer along the feature dim."
                )
            feature_start, feature_stop = variable_slice_config.start, variable_slice_config.stop
            feature_start = feature_start or 0
            feature_stop = feature_stop or layer.shape[-1]

            slice_start = [0 for l in layer.shape[:-1]] + [feature_start]
            slice_size = [-1 for l in layer.shape[:-1]] + [feature_stop-feature_start]

            return tf.slice(layer, slice_start, slice_size)

        else:
            raise KeyError(f"Variable {name} is not in the clip config.")

    def _check_only_feature_dim_is_clipped(self, name):
        if len(self.clip[name]) > 1:
            raise ValueError(
                f"Variable {name} has >1 dimension to clip along. "
                "This method clip_layer_along_feature_dim assumes that only "
                "the feature dim is specified in the clip config."
                )     

def zero_pad_output_feature_dim(layer: tf.Tensor, n_removed_from_start, n_removed_from_end) -> tf.Tensor:
    # Restores a clipped output dim to its original feature dimension size by replacing
    # clipped slices with zeros

    concat_layers = []
    if n_removed_from_start > 0:
        pad_start = _zero_slice_with_placeholder_dim(layer, n_removed_from_start)
        concat_layers.append(pad_start)

    concat_layers.append(layer)

    if n_removed_from_end > 0:
        pad_end = _zero_slice_with_placeholder_dim(layer, n_removed_from_end)
        concat_layers.append(pad_end)
    
    return tf.concat(concat_layers, axis=-1)


def zero_fill_clipped_layers(
        config: ClipConfig, 
        clipped_layers: Sequence[tf.Tensor], 
        variable_names: Sequence[str],
        original_feature_dim_sizes: Sequence[int],
) -> Sequence[tf.Tensor]:
    # Takes layers and restores those that were clipped to their
    # original feature dim sizes by replacing clipped levels with zeroes.
    outputs = []
    for layer, name, original_size in zip(clipped_layers, variable_names, original_feature_dim_sizes):
        if name in config.clip:
            n_removed_from_start, n_removed_from_end = config.removed_sizes(original_size, name)
            outputs.append(zero_pad_output_feature_dim(layer, n_removed_from_start, n_removed_from_end))
        else:
            outputs.append(layer)
    return outputs
    

def clip_layers(
        config: ClipConfig, 
        layers: Sequence[tf.Tensor], 
        variable_names: Sequence[str]
        ) -> Sequence[tf.Tensor]:
    # Takes layers and applies clipping to those that have entries in the ClipConfig.
    # The layer_names input is the list of variable names corresponding
    # to each layer in layers.
    # Returns a sequence of layers of the same length as input layers.
    outputs = []
    for layer, name in zip(layers, variable_names):
        if name in config.clip:
            outputs.append(config.clip_layer_along_feature_dim(layer, name))
        else:
            outputs.append(layer)
    return outputs


def clip_arrays(
        config: ClipConfig, 
        arrays: Sequence[np.ndarray],
        variable_names: Sequence[str],
) -> Sequence[np.ndarray]:
    # Takes np arrays and applies clipping to those that have entries
    # in the ClipConfig. The variable_names input is the list of variable
    # names corresponding to each element in arrays.
    # Returns a sequence of arrays of the same length as input arrays
    # Applies clipping to arrays alont the last (feature) dimension if
    # they have an entry in clip config.
    outputs = []
    for array, name in zip(arrays, variable_names):
        if name in config.clip:
            slice = config.single_feature_slice(name)
            outputs.append(array.take(indices=range(slice.start, slice.stop), axis=-1))
        else:
            outputs.append(array)
    return outputs


def _zero_slice_with_placeholder_dim(layer: tf.Tensor, n_zero_levels: int) -> tf.Tensor:
    # hacky way of getting zero padding tensor with same first placeholder dim as input
    # but different feature dim size
    layer_copy = tf.zeros_like(layer)
    slice_start = [0 for l in layer.shape]
    slice_size = [-1 for l in layer.shape[:-1]] + [n_zero_levels]
    return tf.slice(layer_copy, slice_start, slice_size)

