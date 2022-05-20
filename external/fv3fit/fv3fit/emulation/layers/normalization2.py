"""Normalization transformations
"""
from typing import Optional
from dataclasses import dataclass
import tensorflow as tf
from enum import Enum


class NormLayer(tf.Module):
    """A normalization transform
    """

    def __init__(
        self, scale: tf.Tensor, center: tf.Tensor, name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name)
        self.scale = tf.constant(scale, name=name + "_scale" if name else None)
        self.center = tf.constant(center, name=name + "_center" if name else None)

    def forward(self, tensor: tf.Tensor) -> tf.Tensor:
        return (tensor - self.center) / self.scale

    def backward(self, tensor: tf.Tensor) -> tf.Tensor:
        return tensor * self.scale + self.center


class StdDevMethod(Enum):
    per_feature = "per_feature"
    all = "all"
    max = "max"


class MeanMethod(Enum):
    per_feature = "per_feature"
    all = "all"


@dataclass
class NormFactory:
    scale: StdDevMethod
    center: MeanMethod = MeanMethod.per_feature

    def build(self, sample: tf.Tensor, name: Optional[str] = None) -> NormLayer:
        mean = _compute_center(sample, self.center)
        scale = _compute_scale(sample, self.scale)
        return NormLayer(scale=scale, center=mean, name=name)


def _standard_deviation_all_features(tensor: tf.Tensor) -> tf.Tensor:
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


def _compute_center(tensor: tf.Tensor, method: MeanMethod) -> tf.Tensor:
    fit_center = {
        MeanMethod.per_feature: _fit_mean_per_feature,
        MeanMethod.all: _fit_mean_all,
    }[method]
    return fit_center(tensor)


def _compute_scale(tensor: tf.Tensor, method: StdDevMethod) -> tf.Tensor:
    fit_scale = {
        StdDevMethod.per_feature: _fit_std_per_feature,
        StdDevMethod.all: _standard_deviation_all_features,
        StdDevMethod.max: _fit_std_max,
    }[method]
    return fit_scale(tensor)
