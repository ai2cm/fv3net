import abc
import fsspec
import numpy as np
import os
import tensorflow as tf
from typing import Union, Sequence, Optional, cast
import yaml
import fv3fit
from fv3fit._shared.predictor import Reloadable
from fv3fit.reservoir._reshaping import stack_array_preserving_last_dim

from fv3fit._shared import io


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


@io.register("do-nothing-transformer")
class DoNothingAutoencoder(Transformer, Reloadable):
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
    object and only save output/hybrid transformers if they are
    different from the input transformer.
    """

    INPUT_DIR = "input_transformer"
    OUTPUT_DIR = "output_transformer"
    HYBRID_DIR = "hybrid_transformer"

    def __init__(
        self,
        input: Transformer,
        output: Optional[Transformer] = None,
        hybrid: Optional[Transformer] = None,
    ):
        self.input = input
        self._output_same_as_input = True if output is None else False
        self._hybrid_same_as_input = True if hybrid is None else False
        self.output = output or input
        self.hybrid = hybrid or input

    def dump(self, path):
        self.input.dump(os.path.join(path, self.INPUT_DIR))
        if not self._output_same_as_input:
            self.output.dump(os.path.join(path, self.OUTPUT_DIR))
        if not self._hybrid_same_as_input:
            self.hybrid.dump(os.path.join(path, self.HYBRID_DIR))

    @classmethod
    def load(cls, path) -> "TransformerGroup":
        input = cast(Transformer, fv3fit.load(os.path.join(path, cls.INPUT_DIR)))

        try:
            output: Optional[Transformer] = cast(
                Transformer, fv3fit.load(os.path.join(path, cls.OUTPUT_DIR))
            )
        except (KeyError):
            output = None
        try:
            hybrid: Optional[Transformer] = cast(
                Transformer, fv3fit.load(os.path.join(path, cls.HYBRID_DIR))
            )
        except (KeyError):
            hybrid = None

        return cls(input=input, output=output, hybrid=hybrid)


def decode_columns(
    encoded_output: np.ndarray, transformer: Transformer
) -> Sequence[np.ndarray]:
    """
    Differs from encode_columns as the decoder expects a single input array
    (not a list of one array per variable) and
    can predict multiple outputs rather than a single latent vector.
    Expand a sequnence of N x M x L dim data into i variables
    to one or more N x M x Vi dim array, where Vi is number of features
    (usually vertical levels) of each variable and L << V is a smaller number
    of latent dimensions
    """
    if encoded_output.ndim > 3:
        raise ValueError("Unexpected dimension size in decoding operation.")

    feature_size = encoded_output.shape[-1]
    leading_shape = encoded_output.shape[:-1]
    encoded_output = encoded_output.reshape(-1, feature_size)
    decoded = transformer.decode(encoded_output)
    var_arrays = [arr.reshape(*leading_shape, -1) for arr in decoded]
    return var_arrays


def encode_columns(
    input_arrs: Sequence[tf.Tensor], transformer: Transformer
) -> np.ndarray:
    """
    Reduces a sequnence of N x M x Vi dim data over i variables
    to a single N x M x Z dim array, where Vi is original number of features
    (usually vertical levels) of each variable and Z << V is a smaller number
    of latent dimensions
    """
    original_sample_shape = input_arrs[0].shape[:-1]
    reshaped = [stack_array_preserving_last_dim(var) for var in input_arrs]
    encoded_reshaped = transformer.encode(reshaped)
    return encoded_reshaped.reshape(*original_sample_shape, -1)
