import dataclasses
import numpy as np
import tensorflow as tf
from typing import Hashable, Mapping, Sequence, Union
from fv3fit._shared.config import SliceConfig, PackerConfig

Clippable = Union[tf.Tensor, np.ndarray]


@dataclasses.dataclass(frozen=True)
class ClipConfig(PackerConfig):
    """Config class for implementing input and output clipping in keras models.
    Clips the last dimension, which the user must ensure is the correct dimension.

    Attributes:
        clip: slice of last dimension to retain when clipping, for the given variable
    """

    clip: Mapping[Hashable, SliceConfig] = dataclasses.field(default_factory=dict)

    def _get_mask_array(
        self, unmasked: Union[tf.Tensor, np.ndarray], name: str
    ) -> np.ndarray:
        slice_config = self.clip[name]
        total_length = unmasked.shape[-1]

        start = slice_config.start or 0
        stop = slice_config.stop or total_length
        return np.hstack(
            [np.zeros(start), np.ones(stop - start), np.zeros(total_length - stop)]
        )

    def zero_mask_clipped_layer(self, layer: tf.Tensor, name: str,) -> tf.Tensor:
        """Fills clipped levels with zero, maintaining the original length along the
        clipped dimension. If name is not in config, returns the input layer unchanged.

        Args:
            layer: tensor/layer to clip along last dim
            name: variable name corresponding to entry in ClipConfig.clip
        """
        if name in self.clip:
            mask = self._get_mask_array(layer, name)
            mask_layer = tf.constant(mask, dtype=tf.float32)
            return tf.math.multiply(layer, mask_layer)
        else:
            return layer

    def clip_along_last_dim(self, clip_object: Clippable, name: str) -> Clippable:
        """Clips an array or layer along its last dimension. If name is not in config,
        returns the input clip_object unchanged.

        Args:
            clip_object: np array or tensorflow layer to clip along last dim
            name: variable name corresponding to entry in ClipConfig.clip

        Returns:
            Clipped array or tensorflow layer
        """
        if name in self.clip:
            return clip_object[..., self.clip[name].slice]
        else:
            return clip_object


def clip_sequence(
    config: ClipConfig, clip_objects: Sequence[Clippable], variable_names: Sequence[str]
) -> Sequence[tf.Tensor]:
    """Takes a sequence of arrays or layers and applies clipping to those that have
    entries in the ClipConfig.

    Args:
        config: ClipConfig
        clip_objects: sequence of arrays or layers to clip along last dimension.
        variable_names: ordered list of variable names corresponding to the items
        in sequence
    """
    outputs = []
    for layer, name in zip(clip_objects, variable_names):
        if name in config.clip:
            outputs.append(config.clip_along_last_dim(layer, name))
        else:
            outputs.append(layer)
    return outputs
