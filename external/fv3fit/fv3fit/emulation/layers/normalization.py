import abc
import dataclasses
from typing import Optional
import tensorflow as tf


def standard_deviation_all_features(tensor):
    """Commpute standard deviation across all features.

    A separate mean is computed for each output level.

    Assumes last dimension is feature.
    """
    reduce_axes = tuple(range(len(tensor.shape) - 1))
    mean = tf.cast(tf.reduce_mean(tensor, axis=reduce_axes), tf.float32)
    return tf.cast(tf.sqrt(tf.reduce_mean((tensor - mean) ** 2)), tf.float32,)


class NormLayer(tf.keras.layers.Layer, abc.ABC):
    def __init__(self, name=None, **kwargs):
        super(NormLayer, self).__init__(name=name)
        self.fitted = False

    @abc.abstractmethod
    def _build_mean(self, in_shape):
        self.mean = None

    @abc.abstractmethod
    def _build_sigma(self, in_shape):
        self.sigma = None

    def build(self, in_shape):
        self._build_mean(in_shape)
        self._build_sigma(in_shape)

    @abc.abstractmethod
    def _fit_mean(self, tensor):
        pass

    @abc.abstractmethod
    def _fit_sigma(self, tensor):
        pass

    def fit(self, tensor):
        self(tensor)
        self._fit_mean(tensor)
        self._fit_sigma(tensor)
        self.fitted = True

    @abc.abstractmethod
    def call(self, tensor) -> tf.Tensor:
        pass


class PerFeatureMean(NormLayer):
    """
    Build layer weights and fit a mean value for each
    feature in a tensor (assumed last dimension is feature).
    """

    def _build_mean(self, in_shape):
        self.mean = self.add_weight(
            "mean", shape=[in_shape[-1]], dtype=tf.float32, trainable=False
        )

    def _fit_mean(self, tensor):
        reduce_axes = tuple(range(len(tensor.shape) - 1))
        self.mean.assign(tf.cast(tf.reduce_mean(tensor, axis=reduce_axes), tf.float32))


class PerFeatureStd(NormLayer):
    """
    Build layer weights and fit a standard deviation value [sigma]
    for each feature in a tensor (assumed last dimension is feature).
    """

    def _build_sigma(self, in_shape):
        self.sigma = self.add_weight(
            "sigma", shape=[in_shape[-1]], dtype=tf.float32, trainable=False
        )

    def _fit_sigma(self, tensor):
        reduce_axes = tuple(range(len(tensor.shape) - 1))
        self.sigma.assign(
            tf.cast(tf.math.reduce_std(tensor, axis=reduce_axes), tf.float32)
        )


class FeatureMaxStd(NormLayer):
    """
    Build layer weights and fit a standard deviation value based
    on the maximum of all features in a tensor (assumed last
    dimension is feature).
    """

    def _build_sigma(self, in_shape):
        self.sigma = self.add_weight(
            "sigma", shape=[], dtype=tf.float32, trainable=False
        )

    def _fit_sigma(self, tensor):
        reduce_axes = tuple(range(len(tensor.shape) - 1))
        stddev = tf.math.reduce_std(tensor, axis=reduce_axes)
        max_std = tf.cast(tf.reduce_max(stddev), tf.float32)
        self.sigma.assign(max_std)


class FeatureMeanStd(FeatureMaxStd):
    def _fit_sigma(self, tensor: tf.Tensor) -> None:
        self.sigma.assign(standard_deviation_all_features(tensor))


class StandardNormLayer(PerFeatureMean, PerFeatureStd):
    """
    Normalization layer that removes mean and standard
    deviation for each feature individually.

    Args:
        epsilon: Floating point  floor added to sigma prior to
            division
    """

    def __init__(self, epsilon: float = 1e-7, name=None):
        super().__init__(name=name)
        self.epsilon = epsilon

    def call(self, tensor):
        return (tensor - self.mean) / (self.sigma + self.epsilon)


class StandardDenormLayer(PerFeatureMean, PerFeatureStd):
    """
    De-normalization layer that scales by the standard
    deviation and adds the mean for each feature individually.
    """

    def call(self, tensor):
        return tensor * self.sigma + self.mean


class MaxFeatureStdNormLayer(PerFeatureMean, FeatureMaxStd):
    """
    Normalization layer that removes mean for each feature
    individually but scales all features by the maximum standard
    deviation calculated over all features. Useful to preserve
    feature scale relationships.
    """

    def call(self, tensor):
        return (tensor - self.mean) / self.sigma


class MaxFeatureStdDenormLayer(MaxFeatureStdNormLayer):
    """
    De-normalization layer that scales all features by the maximum
    standard deviation calculated over all features and adds back
    the mean for each individual feature.
    """

    def call(self, tensor):
        return tensor * self.sigma + self.mean


class MeanFeatureStdDenormLayer(MaxFeatureStdDenormLayer, FeatureMeanStd):
    """De-normalization layer that scales all outputs by the standard deviation
    computed from the per-feature mean.
    """

    pass


class MeanFeatureStdNormLayer(MaxFeatureStdNormLayer, FeatureMeanStd):
    """Layer tha normalizes all outputs by the standard deviation computed from
    the per-feature mean.
    """

    pass


@dataclasses.dataclass
class NormalizeConfig:
    """
    Initialize a normalizing layer

    Args:
        class_name: key for layer class passed to `get_norm_class`
        sample_data: sample tensor used to fit the normalizing layer
        layer_name: name for instantiated layer. must be unique if combining
            with other layers into a model
    """

    class_name: str
    sample_data: tf.Tensor
    layer_name: Optional[str] = None

    def initialize_layer(self):
        """Get initialized NormLayer"""
        cls = get_norm_class(self.class_name)
        layer = cls(name=self.layer_name)
        layer.fit(self.sample_data)

        return layer


@dataclasses.dataclass
class DenormalizeConfig:
    """
    Initialize a denormalizing layer

    Args:
        class_name: key for layer class passed to `get_denorm_class`
        sample_data: sample tensor used to fit the denormalizing layer
        layer_name: name for instantiated layer. must be unique if combining
            with other layers into a model
    """

    class_name: str
    sample_data: tf.Tensor
    layer_name: Optional[str] = None

    def initialize_layer(self):
        """Get initialized NormLayer"""
        cls = get_denorm_class(self.class_name)
        layer = cls(name=self.layer_name)
        layer.fit(self.sample_data)

        return layer


MAX_STD = "max_std"
MEAN_STD = "mean_std"
PER_LEVEL_STD = "per_level_std"


def get_norm_class(key):

    if key == MAX_STD:
        return MaxFeatureStdNormLayer
    elif key == MEAN_STD:
        return MeanFeatureStdNormLayer
    elif key == PER_LEVEL_STD:
        return StandardNormLayer
    else:
        raise KeyError(f"Unrecognized normalization layer key provided: {key}")


def get_denorm_class(key):

    if key == MAX_STD:
        return MaxFeatureStdDenormLayer
    elif key == MEAN_STD:
        return MeanFeatureStdDenormLayer
    elif key == PER_LEVEL_STD:
        return StandardDenormLayer
    else:
        raise KeyError(f"Unrecognized de-normalization layer key provided: {key}")
