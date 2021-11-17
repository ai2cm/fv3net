import dataclasses
import numpy as np
import tensorflow as tf
from typing import Mapping, Hashable
from  fv3fit._shared.config import SliceConfig


ClipDims = Mapping[Hashable, Mapping[str, SliceConfig]]


@dataclasses.dataclass(frozen=True)
class ClipConfig:
    clip: ClipDims = dataclasses.field(default_factory=dict)

    def _single_feature_slice(self, name: str) -> SliceConfig:
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
            if len(self.clip[name]) > 1:
                raise ValueError(
                    f"Variable {name} has >1 dimension to clip along. "
                    "This method clip_layer_along_feature_dim assumes that only "
                    "the feature dim is specified in the clip config."
                    )        
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

            return  tf.slice(layer, slice_start, slice_size)

        else:
            raise KeyError(f"Variable {name} is not in the clip config.")


def zero_pad_output_feature_dim(layer: tf.Tensor, n_removed_from_start, n_removed_from_end) -> tf.Tensor:
    # Restores a clipped output dim to its original feature dimension size by replacing
    # clipped slices with zeros
    non_feature_dim_shape = layer.shape[1:-1].as_list()
    start_pad_shape = non_feature_dim_shape + [n_removed_from_start]
    end_pad_shape = non_feature_dim_shape + [n_removed_from_end]

    concat_layers = []
    if n_removed_from_start > 0:
        pad_start = tf.constant([np.zeros(start_pad_shape)], dtype=layer.dtype)
        concat_layers.append(pad_start)

    concat_layers.append(layer)

    if n_removed_from_end > 0:
        pad_end = tf.constant([np.zeros(end_pad_shape)], dtype=layer.dtype)
        concat_layers.append(pad_end)
    
    return tf.concat(concat_layers, axis=-1)
