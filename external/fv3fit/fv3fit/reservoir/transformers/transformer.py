import abc
import numpy as np
import tensorflow as tf
from typing import Union


class Transformer(abc.ABC):
    @property
    @abc.abstractmethod
    def n_latent_dims(self):
        pass

    @abc.abstractmethod
    def encode(self, x: Union[np.ndarray, tf.Tensor]) -> np.ndarray:
        pass

    @abc.abstractmethod
    def decode(self, x: Union[np.ndarray, tf.Tensor]) -> np.ndarray:
        pass
