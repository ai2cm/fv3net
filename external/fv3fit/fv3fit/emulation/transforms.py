import dataclasses
from typing import List, Mapping

import tensorflow as tf
from typing_extensions import Protocol

TensorDict = Mapping[str, tf.Tensor]


class TensorTransform(Protocol):
    def forward(self, x: TensorDict) -> TensorDict:
        pass

    def backward(self, x: TensorDict) -> TensorDict:
        pass


class TransformedLayer(tf.keras.layers.Layer):
    """Transform variables for a model
    """

    def __init__(self, model: tf.keras.layers.Layer, transform: TensorTransform):
        super().__init__()
        self._model = model
        self._transform = transform

    def call(self, x: TensorDict) -> TensorDict:
        return self._transform.backward(self._model(self._transform.forward(x)))


@dataclasses.dataclass
class LogTransform:
    """A univariate transformation for::

        y := log(x  + epsilon)

    Attributes:
        epsilon: the size of the log transform
    """

    epsilon: float = 1e-23

    def forward(self, x: tf.Tensor) -> tf.Tensor:
        return tf.math.log(x + self.epsilon)

    def backward(self, x: tf.Tensor) -> tf.Tensor:
        return tf.math.exp(x) - self.epsilon


@dataclasses.dataclass
class TransformedVariableConfig:
    """A user facing implementation"""

    source: str
    to: str
    transform: LogTransform


class PerVariableTransform(TensorTransform):
    def __init__(self, fields: List[TransformedVariableConfig]):
        self.fields = fields

    def forward(self, x: TensorDict) -> TensorDict:
        out = {**x}
        for transform in self.fields:
            try:
                out[transform.to] = transform.transform.forward(x[transform.source])
            except KeyError:
                pass
        return out

    def backward(self, y: TensorDict) -> TensorDict:
        out = {**y}
        for transform in self.fields:
            try:
                out[transform.source] = transform.transform.backward(y[transform.to])
            except KeyError:
                pass
        return out
