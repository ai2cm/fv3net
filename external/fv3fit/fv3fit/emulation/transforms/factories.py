import dataclasses
from typing import Callable, Sequence, Set

import tensorflow as tf
from fv3fit.emulation.transforms.transforms import (
    ComposedTransform,
    ConditionallyScaledTransform,
    LogTransform,
    TensorTransform,
    UnivariateTransform,
)
from fv3fit.emulation.types import TensorDict
from fv3fit.emulation.zhao_carr_fields import Field
from fv3fit.keras.math import groupby_bins, piecewise
from typing_extensions import Protocol


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


def reduce_std(x: tf.Tensor) -> tf.Tensor:
    mean = tf.reduce_mean(x)
    return tf.sqrt(tf.reduce_mean((x - mean) ** 2))


def fit_conditional(
    x: tf.Tensor, y: tf.Tensor, reduction: Callable[[tf.Tensor], tf.Tensor], bins: int,
) -> Callable[[tf.Tensor], tf.Tensor]:
    # TODO test
    # should work for vals < min and > max
    min = tf.reduce_min(x)
    max = tf.reduce_max(x)
    edges = tf.linspace(min, max, bins + 1)
    values = groupby_bins(edges, x, y, reduction)

    def interp(x: tf.Tensor) -> tf.Tensor:
        return piecewise(edges[:-1], values, x)

    return interp


@dataclasses.dataclass
class ConditionallyScaled(TransformFactory):
    """Conditionally scaled transformation

    Scales the output-input difference of ``field`` by conditional standard
    deviation and mean::

                  d field - E[d field|on]
        to =  --------------------------------
               max[Std[d field|on], min_scale]

    Attributes:
        to: name of the transformed variable.
        condition_on: the variable to condition on
        bins: the number of bins
        field: the prognostic variable spec. The difference between input and
            output data (``d field``) is normalized.
        min_scale: the minimium scale to normalize by. Used when the scale might
            be 0.

    """

    to: str
    condition_on: str
    bins: int
    field: Field
    min_scale: float = 0.0

    def backward_names(self, requested_names: Set[str]) -> Set[str]:
        """List the names needed to compute ``self.to``"""
        if self.to in requested_names:
            return (requested_names - {self.to}) | {
                self.field.input_name,
                self.condition_on,
                self.field.output_name,
            }
        else:
            return requested_names

    def build(self, sample: TensorDict) -> ConditionallyScaledTransform:

        residual_sample = sample[self.field.output_name] - sample[self.field.input_name]

        return ConditionallyScaledTransform(
            to=self.to,
            on=self.condition_on,
            input_name=self.field.input_name,
            output_name=self.field.output_name,
            scale=fit_conditional(
                sample[self.condition_on], residual_sample, reduce_std, self.bins
            ),
            center=fit_conditional(
                sample[self.condition_on], residual_sample, tf.reduce_mean, self.bins
            ),
            min_scale=self.min_scale,
        )


class ComposedTransformFactory(TransformFactory):
    def __init__(self, factories: Sequence[TransformFactory]):
        self.factories = factories

    def backward_names(self, requested_names: Set[str]) -> Set[str]:
        for factory in self.factories:
            requested_names = factory.backward_names(requested_names)
        return requested_names

    def build(self, sample: TensorDict) -> ComposedTransform:
        transforms = [factory.build(sample) for factory in self.factories]
        return ComposedTransform(transforms)
