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

    """

    to: str
    condition_on: str
    source: str
    bins: int
    min_scale: float = 0.0
    fit_filter_magnitude: Optional[float] = None

    def backward_names(self, requested_names: Set[str]) -> Set[str]:
        """List the names needed to compute ``self.to``"""
        new_names = (
            {self.condition_on, self.source} if self.to in requested_names else set()
        )
        return requested_names.union(new_names)

    def build(self, sample: TensorDict) -> ConditionallyScaledTransform:

        if self.fit_filter_magnitude is not None:
            mask = tf.abs(sample[self.source]) > self.fit_filter_magnitude
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

    def _get_first_order_dependencies(self, name: str) -> Set[str]:
        deps: Set[str] = set()
        for factory in self.factories:
            deps_of_name = factory.backward_names({name}) - {name}
            deps = deps.union(deps_of_name)
        return deps

    def backward_names(self, requested_names: Set[str]) -> Set[str]:
        out: Set[str] = set()
        stack = requested_names.copy()
        visited_names = set()
        while stack:
            name = stack.pop()

            if name in visited_names:
                raise ValueError(f"Circular dependency detected for input {name}")

            visited_names.add(name)
            deps = self._get_first_order_dependencies(name)
            if deps == set():
                # name cannot be computed from transforms so we must request it
                # from the data
                out.add(name)
            else:
                # more processing needed
                for dep in deps:
                    stack.add(dep)
        return out

    def build(self, sample: TensorDict) -> ComposedTransform:
        transforms = []
        sample = {**sample}
        for factory in self.factories:
            transform = factory.build(sample)
            sample.update(transform.forward(sample))
            transforms.append(transform)
        return ComposedTransform(transforms)
