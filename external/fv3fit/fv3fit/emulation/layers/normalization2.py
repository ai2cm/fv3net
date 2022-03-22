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
        self,
        scale: tf.Tensor,
        center: tf.Tensor,
        epsilon: float = 0,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name)
        self.scale = tf.Variable(scale, trainable=False)
        self.center = tf.Variable(center, trainable=False)
        self.epsilon = epsilon

    def forward(self, tensor: tf.Tensor) -> tf.Tensor:
        return (tensor - self.center) / (self.center + self.epsilon)

    def backward(self, tensor: tf.Tensor) -> tf.Tensor:
        return tensor * self.scale + self.center


class StdReduction(Enum):
    per_feature = "per_feature"
    all = "all"
    max = "max"


@dataclass
class NormFactory:
    scale: StdReduction

    def build(self, sample: tf.Tensor) -> NormLayer:
        mean = _fit_mean_per_feature(sample)
        scale = _compute_scale(sample, self.scale)

        return NormLayer(scale=scale, center=mean)


def _standard_deviation_all_features(tensor):
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


def _fit_std_per_feature(tensor: tf.Tensor) -> tf.Tensor:
    reduce_axes = tuple(range(len(tensor.shape) - 1))
    return tf.cast(tf.math.reduce_std(tensor, axis=reduce_axes), tf.float32)


def _fit_std_max(tensor: tf.Tensor) -> tf.Tensor:
    reduce_axes = tuple(range(len(tensor.shape) - 1))
    reduce_axes = tuple(range(len(tensor.shape) - 1))
    stddev = tf.math.reduce_std(tensor, axis=reduce_axes)
    max_std = tf.cast(tf.reduce_max(stddev), tf.float32)
    return max_std


def _compute_scale(tensor: tf.Tensor, method: StdReduction):
    if method == StdReduction.per_feature:
        return _fit_std_per_feature(tensor)
    elif method == StdReduction.all:
        return _standard_deviation_all_features(tensor)
    elif method == StdReduction.max:
        return _fit_std_max(tensor)
