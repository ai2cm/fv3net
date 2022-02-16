import dataclasses
from typing import Callable, Optional, Sequence, Set

import tensorflow as tf
from fv3fit.emulation.transforms.transforms import (
    ComposedTransform,
    ConditionallyScaledTransform,
    LogTransform,
    TensorTransform,
    UnivariateTransform,
)
from fv3fit.emulation.types import TensorDict
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
        filter_magnitude: if provided, any values with
            |to-field| < filter_magnitude are removed from the standard
            deviation/mean calculation.

    """

    to: str
    condition_on: str
    source: str
    bins: int
    min_scale: float = 0.0
    filter_magnitude: Optional[float] = None

    def backward_names(self, requested_names: Set[str]) -> Set[str]:
        """List the names needed to compute ``self.to``"""
        new_names = (
            {self.condition_on, self.source} if self.to in requested_names else set()
        )
        return requested_names.union(new_names)

    def build(self, sample: TensorDict) -> ConditionallyScaledTransform:

        if self.filter_magnitude is not None:
            mask = tf.abs(sample[self.source]) > self.filter_magnitude
        else:
            mask = ...

        return ConditionallyScaledTransform(
            to=self.to,
            on=self.condition_on,
            source=self.source,
            scale=fit_conditional(
                sample[self.condition_on][mask],
                sample[self.source][mask],
                reduce_std,
                self.bins,
            ),
            center=fit_conditional(
                sample[self.condition_on][mask],
                sample[self.source][mask],
                tf.reduce_mean,
                self.bins,
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
