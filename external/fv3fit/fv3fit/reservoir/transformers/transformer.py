import abc
import fsspec
import numpy as np
from numpy import ndarray
import os
import tensorflow as tf
from typing import Union, Sequence, cast, Optional
import yaml

import fv3fit
from fv3fit._shared.predictor import Reloadable
from fv3fit._shared import io
from fv3fit.emulation.layers.normalization import (
    NormFactory,
    NormLayer,
    MeanMethod,
    StdDevMethod,
)


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
        super().__init__(**kwargs)

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


@io.register("scale-spatial-concat-z-transformer")
class ScaleSpatialConcatZTransformer(Transformer):
    _CONFIG_NAME = "scale_spatial_concat_z_transformer.yaml"
    _SCALE_NDARRAY = "scale.npy"
    _CENTER_NDARRAY = "center.npy"
    _MASK_NDARRAY = "mask.npy"
    _EPSILON = 1.0e-7

    def __init__(
        self,
        center: np.ndarray,
        scale: np.ndarray,
        spatial_features: Sequence[int],
        num_variables: int,
        mask: Optional[np.ndarray] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._spatial_features = spatial_features
        self._num_variables = num_variables
        self._norm_layer = NormLayer(center=center, scale=scale, epsilon=self._EPSILON)
        self._mask = mask

    @property
    def n_latent_dims(self):
        return self._num_variables * self._spatial_features[-1]

    @property
    def _flat_spatial_len(self):
        return np.product(self._spatial_features)

    @property
    def _stacked_flat_spatial_split_idxs(self):
        return [self._flat_spatial_len * i for i in range(1, self._num_variables)]

    @property
    def _z_dim_split_idxs(self):
        return [self._spatial_features[-1] * i for i in range(1, self._num_variables)]

    def _check_consistent_xyz(self, input_arrs: Sequence[np.ndarray]):
        if len(input_arrs) != self._num_variables:
            raise ValueError(
                f"Expected {self._num_variables} input arrays but got {len(input_arrs)}"
            )

        for i, arr in enumerate(input_arrs):
            if arr.shape[-3:] != self._spatial_features:
                raise ValueError(
                    "All arrays must have the same x,y,z features. "
                    f"Expected {self._spatial_features} but got {arr.shape[-3:]} "
                    f"for array {i}."
                )

    def encode_txyz(self, input_arrs: Sequence[ndarray]) -> ndarray:
        self._check_consistent_xyz(input_arrs)

        leading_dims = input_arrs[0].shape[:-3]
        # stack xyz
        spatial_last_dim = [tf.reshape(arr, (*leading_dims, -1)) for arr in input_arrs]

        # stack all xyz-flattened variables
        stacked_feature = np.concatenate(spatial_last_dim, axis=-1)

        # normalize
        normalized = self.encode(stacked_feature)

        # split xyz-flattened variables
        normalized_arrs = np.split(
            normalized, self._stacked_flat_spatial_split_idxs, axis=-1
        )

        # reshape to xyz and then stack z
        normalized_unstacked = [
            tf.reshape(arr, (*leading_dims, *self._spatial_features))
            for arr in normalized_arrs
        ]
        normalized_stacked_z = np.concatenate(normalized_unstacked, axis=-1)

        if self._mask is not None:
            normalized_stacked_z = normalized_stacked_z * self._mask

        return normalized_stacked_z

    def decode_txyz(self, encoded: ndarray) -> Sequence[ndarray]:
        leading_dims = encoded.shape[:-3]

        if self._mask is not None:
            encoded = encoded * self._mask

        # unstack z
        normalized_arrs = np.split(encoded, self._z_dim_split_idxs, axis=-1)
        self._check_consistent_xyz(normalized_arrs)

        # stack all xyz-flattened variables
        spatial_last_dim = [
            tf.reshape(arr, (*leading_dims, -1)) for arr in normalized_arrs
        ]
        stacked_feature = np.concatenate(spatial_last_dim, axis=-1)

        # denormalize
        unnormalized = self.decode(stacked_feature)

        # split xyz-flattened variables
        unnormalized_arrs = np.split(
            unnormalized, self._stacked_flat_spatial_split_idxs, axis=-1
        )

        # reshape spatial
        original = [
            tf.reshape(arr, (*leading_dims, *self._spatial_features))
            for arr in unnormalized_arrs
        ]
        return original

    def encode(self, input_arr: ndarray) -> ndarray:
        return self._norm_layer.forward(input_arr)

    def decode(self, input_arr: ndarray) -> ndarray:
        return self._norm_layer.backward(input_arr)

    def dump(self, path: str) -> None:
        if self._norm_layer is None:
            raise ValueError("Cannot dump an unbuilt ScaleSpatialConcatZTransformer")

        with fsspec.open(os.path.join(path, self._CONFIG_NAME), "w") as f:
            yaml.dump(
                {
                    "num_variables": self._num_variables,
                    "spatial_features": self._spatial_features,
                },
                f,
            )

        np.save(os.path.join(path, self._SCALE_NDARRAY), self._norm_layer.scale)
        np.save(os.path.join(path, self._CENTER_NDARRAY), self._norm_layer.center)
        if self._mask is not None:
            np.save(os.path.join(path, self._MASK_NDARRAY), self._mask)

    @classmethod
    def load(cls, path: str) -> "ScaleSpatialConcatZTransformer":
        with fsspec.open(os.path.join(path, cls._CONFIG_NAME), "r") as f:
            config = yaml.safe_load(f)

        scale = np.load(os.path.join(path, cls._SCALE_NDARRAY))
        center = np.load(os.path.join(path, cls._CENTER_NDARRAY))
        if os.path.exists(os.path.join(path, cls._MASK_NDARRAY)):
            mask = np.load(os.path.join(path, cls._MASK_NDARRAY))
        else:
            mask = None
        return cls(center=center, scale=scale, mask=mask, **config)


def build_scale_spatial_concat_z_transformer(
    sample_data: Sequence[np.ndarray], mask: Optional[np.ndarray] = None,
):
    """
    Take in a sequence of time xyz data and form a standard normalizer
    over each xyz element
    """
    leading_dims = sample_data[0].shape[:-3]
    spatial_features = sample_data[0].shape[-3:]
    num_variables = len(sample_data)

    spatial_stacked = [arr.reshape(*leading_dims, -1) for arr in sample_data]
    joined_feature = np.concatenate(spatial_stacked, axis=-1)
    factory = NormFactory(
        center=MeanMethod.per_feature,
        scale=StdDevMethod.per_feature,
        epsilon=ScaleSpatialConcatZTransformer._EPSILON,
    )
    norm_layer = factory.build(joined_feature,)

    return ScaleSpatialConcatZTransformer(
        center=norm_layer.center,
        scale=norm_layer.scale,
        spatial_features=spatial_features,
        num_variables=num_variables,
        mask=mask,
    )

    # also include a mask mapping? that will selectively apply the mask to the fields
    # that way I can mask the SST data and/or the atmospheric data
    # I need a way to turn off the mask application on the encoded output in training


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
