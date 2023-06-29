import abc
import numpy as np
import tensorflow as tf
from typing import Union, Sequence

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


def encode_columns(data: Sequence[tf.Tensor], transformer: Transformer) -> np.ndarray:
    # reduce a sequnence of N x M x Vi dim data over i variables
    # to a single N x M x Z dim array, where Vi is original number of features
    # (usually vertical levels) of each variable and Z << V is a smaller number
    # of latent dimensions
    original_sample_shape = data[0].shape[:-1]
    reshaped = [stack_array_preserving_last_dim(var) for var in data]
    encoded_reshaped = transformer.encode(reshaped)
    return encoded_reshaped.reshape(*original_sample_shape, -1)
