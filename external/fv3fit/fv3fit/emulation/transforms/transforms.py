import dataclasses
from typing import List

import tensorflow as tf
from typing_extensions import Protocol
from fv3fit.emulation.types import TensorDict


class TensorTransform(Protocol):
    def forward(self, x: TensorDict) -> TensorDict:
        pass

    def backward(self, y: TensorDict) -> TensorDict:
        pass


@dataclasses.dataclass
class LogTransform:
    """A univariate transformation for::

        y := log(max(x,epsilon))
        x : = exp(x)

    This is not strictly a bijection because of the quashing at epsilon.

    Attributes:
        epsilon: the size of the log transform
    """

    epsilon: float = 1e-30

    def forward(self, x: tf.Tensor) -> tf.Tensor:
        return tf.math.log(tf.maximum(x, self.epsilon))

    def backward(self, x: tf.Tensor) -> tf.Tensor:
        return tf.math.exp(x)


class UnivariateTransform(TensorTransform):
    def __init__(self, source: str, to: str, transform: LogTransform):
        self.source = source
        self.to = to
        self.transform = transform

    def forward(self, x: TensorDict) -> TensorDict:
        out = {**x}
        out[self.to] = self.transform.forward(x[self.source])
        return out

    def backward(self, y: TensorDict) -> TensorDict:
        out = {**y}
        out[self.source] = self.transform.backward(y[self.to])
        return out


class ComposedTransform(TensorTransform):
    def __init__(self, transforms: List[TensorTransform]):
        self.transforms = transforms

    def forward(self, x: TensorDict) -> TensorDict:
        for transform in self.transforms:
            try:
                x = transform.forward(x)
            except KeyError:
                pass
        return x

    def backward(self, y: TensorDict) -> TensorDict:
        for transform in self.transforms:
            try:
                y = transform.backward(y)
            except KeyError:
                pass
        return y


Identity = ComposedTransform([])
