import abc
import fsspec
import numpy as np
import os
import tensorflow as tf
from typing import Union, Sequence, cast
import yaml

import fv3fit
from fv3fit._shared.predictor import Reloadable
from fv3fit._shared import io

# from fv3fit.emulation.layers.normalization import (
#     NormFactory,
#     NormLayer,
#     MeanMethod,
#     StdDevMethod,
# )


ArrayLike = Union[np.ndarray, tf.Tensor]


class BaseTransformer(abc.ABC):
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


class Transformer(BaseTransformer, Reloadable):
    def __init__(self, **kwargs):
        self.super().__init__(**kwargs)

    def encode_txyz(self, input_arrs: Sequence[np.ndarray]) -> np.ndarray:
        """Handle non-2D inputs during runtime/training"""
        leading_shape = input_arrs[0].shape[:-1]
        collapsed_arrs = [np.reshape(arr, (-1, arr.shape[-1])) for arr in input_arrs]
        encoded = self.encode(collapsed_arrs)
        return np.reshape(encoded, (*leading_shape, -1))

    def decode_txyz(self, encoded: np.ndarray) -> Sequence[np.ndarray]:
        """Handle non-2D inputs during runtime/training"""
        feature_size = encoded.shape[-1]
        leading_shape = encoded.shape[:-1]
        encoded = encoded.reshape(-1, feature_size)
        decoded = self.decode(encoded)
        var_arrays = [tf.reshape(arr, (*leading_shape, -1)) for arr in decoded]
        return var_arrays


@io.register("do-nothing-transformer")
class DoNothingAutoencoder(Transformer):
    _CONFIG_NAME = "mock_transformer.yaml"

    """Useful class for tests. Encode just concatenates input
    variables. Decode separates them back into individual arrays.
    """

    def __init__(self, original_feature_sizes: Sequence[int]):
        self.original_feature_sizes = original_feature_sizes

    @property
    def n_latent_dims(self):
        return sum(self.original_feature_sizes)

    def encode(self, x):
        self.original_feature_sizes = [arr.shape[-1] for arr in x]
        return np.concatenate(x, -1)

    def decode(self, latent_x):
        if self.original_feature_sizes is None:
            raise ValueError("Must encode data before decoding.")

        split_indices = np.cumsum(self.original_feature_sizes)[:-1]
        return np.split(latent_x, split_indices, axis=-1)

    def dump(self, path: str) -> None:
        with fsspec.open(os.path.join(path, self._CONFIG_NAME), "w") as f:
            yaml.dump(
                {"original_feature_sizes": self.original_feature_sizes}, f,
            )

    @classmethod
    def load(cls, path: str) -> "DoNothingAutoencoder":
        with fsspec.open(os.path.join(path, cls._CONFIG_NAME), "r") as f:
            config = yaml.safe_load(f)
        return cls(original_feature_sizes=config["original_feature_sizes"],)


class TransformerGroup:
    """For convenience, keep all the transformers together in a single
    object. To streamline the logic, there may be replicated transformers
    stored when variable groups are identical sets.
    """

    INPUT_DIR = "input_transformer"
    OUTPUT_DIR = "output_transformer"
    HYBRID_DIR = "hybrid_transformer"

    def __init__(self, input: Transformer, output: Transformer, hybrid: Transformer):
        self.input = input
        self.output = output
        self.hybrid = hybrid

    def dump(self, path):

        self.input.dump(os.path.join(path, self.INPUT_DIR))
        self.output.dump(os.path.join(path, self.OUTPUT_DIR))
        self.hybrid.dump(os.path.join(path, self.HYBRID_DIR))

    @classmethod
    def load(cls, path) -> "TransformerGroup":
        input = cast(Transformer, fv3fit.load(os.path.join(path, cls.INPUT_DIR)))
        output = cast(Transformer, fv3fit.load(os.path.join(path, cls.OUTPUT_DIR)))
        hybrid = cast(Transformer, fv3fit.load(os.path.join(path, cls.HYBRID_DIR)))
        return cls(input=input, output=output, hybrid=hybrid)
