import abc
import fsspec
import numpy as np
import os
import tensorflow as tf
from typing import Union, Sequence, Optional
import yaml
from fv3fit._shared.predictor import Reloadable
from fv3fit.reservoir._reshaping import stack_array_preserving_last_dim

from fv3fit._shared import io


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

    def __init__(
        self, latent_dim_len, original_feature_sizes: Optional[Sequence[int]] = None
    ):
        self._latent_dim_len = latent_dim_len
        self.original_feature_sizes = original_feature_sizes

    @property
    def n_latent_dims(self):
        return self._latent_dim_len

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
                {
                    "latent_dim_len": self.n_latent_dims,
                    "original_feature_sizes": self.original_feature_sizes,
                },
                f,
            )

    @classmethod
    def load(cls, path: str) -> "DoNothingAutoencoder":
        with fsspec.open(os.path.join(path, cls._CONFIG_NAME), "r") as f:
            config = yaml.safe_load(f)
        return cls(
            latent_dim_len=config["latent_dim_len"],
            original_feature_sizes=config["original_feature_sizes"],
        )


def decode_columns(
    encoded_output: np.ndarray, transformer: Transformer, xy_shape: Sequence[int]
) -> Sequence[np.ndarray]:
    # Differs from encode_columns as the decoder expects a single input array
    # (not a list of one array per variable) and
    # can predict multiple outputs rather than a single latent vector.
    # Expand a sequnence of N x M x L dim data into i variables
    # to one or more N x M x Vi dim array, where Vi is number of features
    # (usually vertical levels) of each variable and L << V is a smaller number
    # of latent dimensions
    if encoded_output.ndim > 3:
        raise ValueError("Unexpected dimension size in decoding operation.")

    feature_size = encoded_output.shape[-1]
    encoded_output = encoded_output.reshape(-1, feature_size)
    decoded = transformer.decode(encoded_output)
    var_arrays = [arr.reshape(*xy_shape, -1) for arr in decoded]
    return var_arrays


def encode_columns(
    input_arrs: Sequence[tf.Tensor], transformer: Transformer
) -> np.ndarray:
    # reduce a sequnence of N x M x Vi dim data over i variables
    # to a single N x M x Z dim array, where Vi is original number of features
    # (usually vertical levels) of each variable and Z << V is a smaller number
    # of latent dimensions
    original_sample_shape = input_arrs[0].shape[:-1]
    reshaped = [stack_array_preserving_last_dim(var) for var in input_arrs]
    encoded_reshaped = transformer.encode(reshaped)
    return encoded_reshaped.reshape(*original_sample_shape, -1)
