"""Normalization transformations
"""
from typing import Optional
from dataclasses import dataclass
import tensorflow as tf
from enum import Enum


class NormLayer(tf.Module):
    """
    A normalization transform that provides both forward and backward
    transformations. Note that if epslion is not None, the forward
    transformation will be scaled by epsilon, which is useful for
    backwards compatibility with previous usage of StandardNormLayer.

    In the future, we should probably make forward and backward
    transformations use epsilon so it doesn't adjust the round trip
    scaling.
    """

    def __init__(
        self,
        scale: tf.Tensor,
        center: tf.Tensor,
        name: Optional[str] = None,
        epsilon: Optional[float] = None,
    ) -> None:
        super().__init__(name=name)

        self.scale = tf.constant(
            tf.cast(scale, tf.float32), name=name + "_scale" if name else None
        )
        self.center = tf.constant(
            tf.cast(center, tf.float32), name=name + "_center" if name else None
        )

        # For backwards compatibilty w/ tests, we want the option to use no epsilon
        # adjustment.  tf.cast doesn't work when sybolic tensors w/ no type are provided
        # to certain layers that use normalization.
        if epsilon is None:
            self._forward_scale = self.scale
        else:
            self._forward_scale = self.scale + tf.cast(epsilon, tf.float32)

    def forward(self, tensor: tf.Tensor) -> tf.Tensor:
        return (tensor - self.center) / self._forward_scale

    def backward(self, tensor: tf.Tensor) -> tf.Tensor:
        return tensor * self.scale + self.center


class StdDevMethod(Enum):
    """
    per_feature: std calculated at each individual feature
    all: std calculated over all features, but the centering in the calculation
        is per feature
    all_center_all: std calculated over all features (mean also calculated over
        all features)
    max: std is the max of the per_feature calculated std values
    mean: std is the mean of the per_feature calculated std values
    none: std is 1
    """

    per_feature = "per_feature"
    all = "all"
    all_center_all = "all_center_all"
    max = "max"
    mean = "mean"
    none = "none"


class MeanMethod(Enum):
    """
    per_feature: mean calculated for each feature
    all: mean calculated over all features
    none: mean is 0
    """

    per_feature = "per_feature"
    all = "all"
    none = "none"


@dataclass
class NormFactory:
    scale: StdDevMethod
    center: MeanMethod = MeanMethod.per_feature
    epsilon: Optional[float] = None

    def build(self, sample: tf.Tensor, name: Optional[str] = None) -> NormLayer:
        mean = _compute_center(sample, self.center)
        scale = _compute_scale(sample, self.scale)
        return NormLayer(scale=scale, center=mean, name=name, epsilon=self.epsilon,)


def norm2_factory_from_key(key):
    if key == "max_std":
        return NormFactory(StdDevMethod.max, MeanMethod.per_feature,)
    elif key == "mean_std":
        return NormFactory(StdDevMethod.all, MeanMethod.per_feature,)
    else:
        raise KeyError(f"Unrecognized normalization layer key provided: {key}")


def standard_deviation_all_features(tensor: tf.Tensor) -> tf.Tensor:
    """Commpute standard deviation across all features.

    A separate mean is computed for each output level.

    Assumes last dimension is feature.
    """
    reduce_axes = tuple(range(len(tensor.shape) - 1))
    mean = tf.cast(tf.reduce_mean(tensor, axis=reduce_axes), tf.float32)
    return tf.cast(tf.sqrt(tf.reduce_mean((tensor - mean) ** 2)), tf.float32,)


def _fit_mean_per_feature(tensor: tf.Tensor) -> tf.Tensor:
    reduce_axes = tuple(range(len(tensor.shape) - 1))
    return tf.cast(tf.reduce_mean(tensor, axis=reduce_axes), tf.float32)


def _fit_mean_all(tensor: tf.Tensor) -> tf.Tensor:
    return tf.cast(tf.reduce_mean(tensor), tf.float32)


def _fit_std_per_feature(tensor: tf.Tensor) -> tf.Tensor:
    reduce_axes = tuple(range(len(tensor.shape) - 1))
    return tf.cast(tf.math.reduce_std(tensor, axis=reduce_axes), tf.float32)


def _fit_std_max(tensor: tf.Tensor) -> tf.Tensor:
    reduce_axes = tuple(range(len(tensor.shape) - 1))
    reduce_axes = tuple(range(len(tensor.shape) - 1))
    stddev = tf.math.reduce_std(tensor, axis=reduce_axes)
    max_std = tf.cast(tf.reduce_max(stddev), tf.float32)
    return max_std


def _fit_std_mean(tensor: tf.Tensor) -> tf.Tensor:
    reduce_axes = tuple(range(len(tensor.shape) - 1))
    stddev = tf.math.reduce_std(tensor, axis=reduce_axes)
    mean_std = tf.cast(tf.reduce_mean(stddev), tf.float32)
    return mean_std


def _compute_center(tensor: tf.Tensor, method: MeanMethod) -> tf.Tensor:
    fit_center = {
        MeanMethod.per_feature: _fit_mean_per_feature,
        MeanMethod.all: _fit_mean_all,
        MeanMethod.none: lambda _: tf.constant(0, dtype=tf.float32),
    }[method]
    return fit_center(tensor)


def _compute_scale(tensor: tf.Tensor, method: StdDevMethod) -> tf.Tensor:
    fit_scale = {
        StdDevMethod.per_feature: _fit_std_per_feature,
        StdDevMethod.all: standard_deviation_all_features,
        StdDevMethod.all_center_all: tf.math.reduce_std,
        StdDevMethod.max: _fit_std_max,
        StdDevMethod.mean: _fit_std_mean,
        StdDevMethod.none: lambda _: tf.constant(1, dtype=tf.float32),
    }[method]
    return fit_scale(tensor)
