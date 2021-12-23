import dataclasses
from typing import List, Set

import tensorflow as tf
from typing_extensions import Protocol
from fv3fit.emulation.types import TensorDict


class TensorTransform(Protocol):
    def forward(self, x: TensorDict) -> TensorDict:
        pass

    def backward(self, x: TensorDict) -> TensorDict:
        pass

    def backward_names(self, requested_names: Set[str]) -> Set[str]:
        """input names needed to compute the requested transformed names

        Needed for data loading purposes.
        
        """
        pass


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

    def backward_names(self, requested_names: Set[str]) -> Set[str]:
        input_by_output_name = {field.to: field.source for field in self.fields}
        return {input_by_output_name.get(name, name) for name in requested_names}


Identity = PerVariableTransform([])
