import abc
import tensorflow as tf


class NormLayer(tf.keras.layers.Layer, abc.ABC):

    def __init__(self, name=None, **kwargs):
        super(NormLayer, self).__init__(name=name)

    @abc.abstractmethod
    def _build_mean(self, in_shape):
        self.mean = None

    @abc.abstractmethod
    def _build_sigma(self, in_shape):
        self.sigma = None

    def build(self, in_shape):
        self._build_mean(in_shape)
        self._build_sigma(in_shape)
        self.fitted = False

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


class PerLevelMean(NormLayer):

    def _build_mean(self, in_shape):
        self.mean = self.add_weight(
            "mean", shape=[in_shape[-1]], dtype=tf.float32, trainable=False
        )

    def _fit_mean(self, tensor):
        self.mean.assign(tf.cast(tf.reduce_mean(tensor, axis=0), tf.float32))


class PerLevelStd(NormLayer):

    def _build_sigma(self, in_shape):
        self.sigma = self.add_weight(
            "sigma", shape=[in_shape[-1]], dtype=tf.float32, trainable=False
        )

    def _fit_sigma(self, tensor):
        self.sigma.assign(
            tf.cast(tf.math.reduce_std(tensor, axis=0), tf.float32)
        )


class LevelMaxStd(NormLayer):

    def _build_sigma(self, in_shape):
        self.sigma = self.add_weight(
            "sigma", shape=[], dtype=tf.float32, trainable=False
        )

    def _fit_sigma(self, tensor):
        stddev = tf.math.reduce_std(tensor, axis=0)
        max_std = tf.cast(tf.reduce_max(stddev), tf.float32)
        self.sigma.assign(max_std)


class StandardNormLayer(PerLevelMean, PerLevelStd):
    def __init__(self, epsilon: float = 1e-7, name=None):
        super(NormLayer, self).__init__(name=name)
        self.epsilon = epsilon

    def call(self, tensor):
        return (tensor - self.mean) / (self.sigma + self.epsilon)


class StandardDenormLayer(StandardNormLayer):

    def call(self, tensor):
        return tensor * self.sigma + self.mean


class MaxProfileStdNormLayer(PerLevelMean, LevelMaxStd):

    def call(self, tensor):
        return (tensor - self.mean) / self.sigma


class MaxProfileStdDenormLayer(MaxProfileStdNormLayer):

    def call(self, tensor):
        return tensor * self.sigma + self.mean
