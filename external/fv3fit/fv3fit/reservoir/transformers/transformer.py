import abc
import fsspec
import numpy as np
import os
import tensorflow as tf
from typing import Union, Sequence
import yaml
from fv3fit._shared.predictor import Reloadable

from fv3fit._shared import io
from fv3fit.reservoir._reshaping import stack_array_preserving_last_dim


ArrayLike = Union[np.ndarray, tf.Tensor]


class Transformer(abc.ABC):
    @property
    @abc.abstractmethod
    def n_latent_dims(self):
        pass

    @abc.abstractmethod
    def encode(self, x: Sequence[ArrayLike]) -> np.ndarray:
        pass

    @abc.abstractmethod
    def decode(self, x: ArrayLike) -> Sequence[ArrayLike]:
        pass


@io.register("do-nothing-transformer")
class DoNothingAutoencoder(Transformer, Reloadable):
    _CONFIG_NAME = "mock_transformer.yaml"

    """Useful class for tests. Encode just concatenates input
    variables. Decode separates them back into individual arrays.
    """

    def __init__(self, latent_dim_len):
        self._latent_dim_len = latent_dim_len
        self._array_feature_sizes = None

    @property
    def n_latent_dims(self):
        return self._latent_dim_len

    def encode(self, x):
        self._array_feature_sizes = [arr.shape[-1] for arr in x]
        return np.concatenate(x, -1)

    def decode(self, latent_x):
        if self._array_feature_sizes is None:
            raise ValueError("Must encode data before decoding.")

        split_indices = np.cumsum(self._array_feature_sizes)[:-1]
        return np.split(latent_x, split_indices, axis=-1)

    def dump(self, path: str) -> None:
        with fsspec.open(os.path.join(path, self._CONFIG_NAME), "w") as f:
            yaml.dump({"n_latent_dims": self.n_latent_dims}, f)

    @classmethod
    def load(cls, path: str) -> "DoNothingAutoencoder":
        with fsspec.open(os.path.join(path, cls._CONFIG_NAME), "r") as f:
            config = yaml.safe_load(f)
        return cls(config["n_latent_dims"])


def encode_columns(data: Sequence[tf.Tensor], transformer: Transformer) -> np.ndarray:
    # reduce a sequnence of N x M x Vi dim data over i variables
    # to a single N x M x Z dim array, where Vi is original number of features
    # (usually vertical levels) of each variable and Z << V is a smaller number
    # of latent dimensions
    original_sample_shape = data[0].shape[:-1]
    reshaped = [stack_array_preserving_last_dim(var) for var in data]
    encoded_reshaped = transformer.encode(reshaped)
    return encoded_reshaped.reshape(*original_sample_shape, -1)


def _encode_input_variables(input_arrs: Sequence[tf.Tensor], transformer: Transformer):
    sample_dims_shape = list(input_arrs[0].shape[:-1])
    feature_len = input_arrs[0].shape[-1]
    stacked_sample_arrs = [arr.reshape(-1, feature_len) for arr in input_arrs]

    encoded = transformer.encode(stacked_sample_arrs)
    encoded_shape = sample_dims_shape + [encoded.shape[-1]]
    return encoded.reshape(encoded_shape)
