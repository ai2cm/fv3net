from typing import Sequence, Mapping
from fv3fit._shared.packer import ArrayPacker
import tensorflow as tf


class Unpack(tf.keras.layers.Layer):
    """
    Layer which unpacks a stacked tensor into a sequence of unstacked tensors
    """

    def __init__(
        self,
        *,
        pack_names: Sequence[str],
        n_features: Mapping[str, int],
        feature_dim: int,
    ):
        """
        Args:
            pack_names: variable names in order
            n_features: indicates the length of the feature dimension for each variable
            feature_dim: the axis of the feature dimension, must be 1 or 2
        """
        super().__init__()
        self.pack_names = pack_names
        self.n_features = n_features
        self.feature_dim = feature_dim
        if feature_dim not in (1, 2):
            raise NotImplementedError(self.feature_dim)

    def call(self, inputs):
        i = 0
        return_tensors = []
        for name in self.pack_names:
            features = self.n_features[name]
            if self.feature_dim == 1:
                return_tensors.append(inputs[:, i : i + features])
            elif self.feature_dim == 2:
                return_tensors.append(inputs[:, :, i : i + features])
            else:
                raise NotImplementedError(self.feature_dim)
            i += features
        return return_tensors

    def get_config(self):
        return {
            "pack_names": self.pack_names,
            "n_features": self.n_features,
            "feature_dim": self.feature_dim,
        }


class LayerPacker(ArrayPacker):
    def pack_layer(self):
        if len(self.pack_names) > 1:
            return tf.keras.layers.Concatenate()
        else:
            raise NotImplementedError(
                "pack layer only implemented for multiple pack variables, "
                "avoid adding a pack layer when len(obj.pack_names) is 1"
            )

    def unpack_layer(self, feature_dim: int):
        # have to store this as a local scope variable
        # so that serialization does not cause self to be serialized
        return Unpack(
            pack_names=self.pack_names,
            n_features=self._n_features,
            feature_dim=feature_dim,
        )
