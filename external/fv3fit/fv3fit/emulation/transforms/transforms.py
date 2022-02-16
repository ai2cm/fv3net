import dataclasses
from typing import Callable, List, Set

import tensorflow as tf
from typing_extensions import Protocol
from fv3fit.emulation.types import TensorDict


class TensorTransform(Protocol):
    def forward(self, x: TensorDict) -> TensorDict:
        pass

    def backward(self, y: TensorDict) -> TensorDict:
        pass


@dataclasses.dataclass
class Difference(TensorTransform):
    """A difference variable::

        to = after - before

    """

    to: str
    before: str
    after: str

    def backward_names(self, requested_names: Set[str]) -> Set[str]:
        new_names = {self.before, self.after} if self.to in requested_names else set()
        return requested_names.union(new_names)

    def build(self, sample: TensorDict) -> TensorTransform:
        return self

    def forward(self, x: TensorDict) -> TensorDict:
        x = {**x}
        x[self.to] = x[self.after] - x[self.before]
        return x

    def backward(self, y: TensorDict) -> TensorDict:
        y = {**y}
        y[self.after] = y[self.before] + y[self.to]
        return y


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


class ConditionallyScaledTransform(TensorTransform):
    def __init__(
        self,
        to: str,
        input_name: str,
        output_name: str,
        on: str,
        scale: Callable[[tf.Tensor], tf.Tensor],
        center: Callable[[tf.Tensor], tf.Tensor],
        min_scale: float = 0.0,
    ) -> None:
        self.to = to
        self.input_name = input_name
        self.output_name = output_name
        self.on = on
        self.scale = scale
        self.center = center
        self.min_scale = min_scale

    def _limited_scale(self, x: tf.Tensor) -> tf.Tensor:
        return tf.maximum(self.scale(x), self.min_scale)

    def forward(self, x: TensorDict) -> TensorDict:
        out = {**x}
        out[self.to] = (
            x[self.output_name] - x[self.input_name] - self.center(x[self.on])
        ) / self._limited_scale(x[self.on])
        return out

    def backward(self, y: TensorDict) -> TensorDict:
        out = {**y}
        out[self.output_name] = (
            y[self.to] * self._limited_scale(y[self.on])
            + y[self.input_name]
            + self.center(y[self.on])
        )
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
        for transform in self.transforms[::-1]:
            try:
                y = transform.backward(y)
            except KeyError:
                pass
        return y


Identity = ComposedTransform([])
