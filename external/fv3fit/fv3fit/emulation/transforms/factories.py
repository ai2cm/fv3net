import dataclasses
from typing import Callable, Optional, Sequence, Set

import tensorflow as tf
from fv3fit.emulation.transforms.transforms import (
    ComposedTransform,
    ConditionallyScaledTransform,
    TensorTransform,
    UnivariateCompatible,
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
    transform: UnivariateCompatible
    to: Optional[str] = None

    def backward_names(self, requested_names: Set[str]) -> Set[str]:
        if self.to in requested_names:
            return (requested_names - {self.to}) | {self.source}
        else:
            return requested_names

    def build(self, sample: TensorDict) -> UnivariateTransform:
        return UnivariateTransform(self.source, self.transform, to=self.to)


def fit_conditional(
    x: tf.Tensor,
    y: tf.Tensor,
    reduction: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
    bins: int,
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

    Scales ``source`` by conditional standard deviation and mean::

                  source - E[source|on]
        to =  --------------------------------
               max[Std[source|on], min_scale]

    Attributes:
        to: name of the transformed variable.
        condition_on: the variable to condition on
        bins: the number of bins
        source: The variable to be normalized.
        min_scale: the minimium scale to normalize by. Used when the scale might
            be 0.
        fit_filter_magnitude: if provided, any values with
            |source| < filter_magnitude are removed from the standard
            deviation/mean calculation.
        weights: name of variable to weight the expectations by

    """

    to: str
    condition_on: str
    source: str
    bins: int
    min_scale: float = 0.0
    fit_filter_magnitude: Optional[float] = None
    weights: str = ""

    def backward_names(self, requested_names: Set[str]) -> Set[str]:
        """List the names needed to compute ``self.to``"""

        if self.to in requested_names:
            dependencies = {self.condition_on, self.source}
            requested_names = (requested_names - {self.to}) | dependencies
            if self.weights:
                requested_names.add(self.weights)

        return requested_names

    def build(self, sample: TensorDict) -> ConditionallyScaledTransform:

        if self.fit_filter_magnitude is not None:
            weights = tf.abs(sample[self.source]) > self.fit_filter_magnitude
        else:
            weights = tf.ones_like(sample[self.source], dtype=tf.bool)

        if self.weights:
            weights *= sample[self.weights]

        def weighted_mean(x, w):
            dt = x.dtype
            w_float = tf.cast(w, dt) * tf.cast(weights, dt)
            return tf.reduce_sum(w_float * x) / tf.reduce_sum(w_float)

        def weighted_std(x, w):
            mu = weighted_mean(x, w)
            return tf.math.sqrt(weighted_mean((x - mu) ** 2, w))

        return ConditionallyScaledTransform(
            to=self.to,
            on=self.condition_on,
            source=self.source,
            scale=fit_conditional(
                sample[self.condition_on], sample[self.source], weighted_std, self.bins,
            ),
            center=fit_conditional(
                sample[self.condition_on],
                sample[self.source],
                weighted_mean,
                self.bins,
            ),
            min_scale=self.min_scale,
        )


class ComposedTransformFactory(TransformFactory):
    def __init__(self, factories: Sequence[TransformFactory]):
        self.factories = factories

    def backward_names(self, requested_names: Set[str]) -> Set[str]:
        for factory in self.factories[::-1]:
            requested_names = factory.backward_names(requested_names)
        return requested_names

    def build(self, sample: TensorDict) -> ComposedTransform:
        transforms = []
        sample = {**sample}
        for factory in self.factories:
            transform = factory.build(sample)
            sample.update(transform.forward(sample))
            transforms.append(transform)
        return ComposedTransform(transforms)
