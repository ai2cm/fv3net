import dataclasses
from typing import Sequence, Set

from typing_extensions import Protocol
from fv3fit.emulation.types import TensorDict
from fv3fit.emulation.transforms.transforms import (
    TensorTransform,
    LogTransform,
    UnivariateTransform,
    ComposedTransform,
)


class TransformFactory(Protocol):
    """The interface of a static configuration object

    Methods:
        backward_names: used to infer to the required input variables
        build: builds the transform given some data

    """

    def backward_names(self, requested_names: Set[str]) -> Set[str]:
        pass

    def build(self, sample: TensorDict) -> TensorTransform:
        pass


@dataclasses.dataclass
class TransformedVariableConfig(TransformFactory):
    """A user facing implementation"""

    source: str
    to: str
    transform: LogTransform

    def backward_names(self, requested_names: Set[str]) -> Set[str]:
        if self.to in requested_names:
            return (requested_names - {self.to}) | {self.source}
        else:
            return requested_names

    def build(self, sample: TensorDict) -> TensorTransform:
        return UnivariateTransform(self.source, self.to, self.transform)


class ComposedTransformFactory(TransformFactory):
    def __init__(self, factories: Sequence[TransformFactory]):
        self.factories = factories

    def backward_names(self, requested_names: Set[str]) -> Set[str]:
        for factory in self.factories:
            requested_names = factory.backward_names(requested_names)
        return requested_names

    def build(self, sample: TensorDict) -> TensorTransform:
        transforms = [factory.build(sample) for factory in self.factories]
        return ComposedTransform(transforms)
