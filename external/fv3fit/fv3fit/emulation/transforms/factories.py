import dataclasses
from typing import Callable, List, Optional, Sequence, Set

import tensorflow as tf
from fv3fit.emulation.transforms.transforms import (
    ComposedTransform,
    ConditionallyScaledTransform,
    LogTransform,
    TensorTransform,
    UnivariateTransform,
    DifferenceTransform,
)
from fv3fit.emulation.types import TensorDict
from fv3fit.keras.math import groupby_bins, piecewise
from typing_extensions import Protocol


def _get_dependencies(name: str, factories: Sequence["TransformFactory"]):

    deps = set()
    intermediate_deps = set()

    # Traverse backwards through transform for requested name
    for i, factory in enumerate(factories[::-1], 1):

        if factory.to == name:

            intermediate_deps |= factory.required_names

            # retrieve dependencies for each earlier in pipeline
            for dep_name in sorted(factory.required_names):

                new_deps, intermediate = _get_dependencies(dep_name, factories[:-i])
                deps |= new_deps
                intermediate_deps |= intermediate

            break
    else:
        return {name}, set()

    return deps, intermediate_deps - deps


class TransformFactory(Protocol):
    """The interface of a static configuration object

    Methods:
        backward_names: used to infer to the required input variables
        build: builds the transform given some data
        required_names: required inputs for successful transform

    """

    to: str

    def backward_names(self, requested_names: Set[str]) -> Set[str]:
        pass

    def build(self, sample: TensorDict) -> TensorTransform:
        pass

    @property
    def required_names(self) -> Set[str]:
        return set()

    @property
    def _factory_list(self) -> Sequence["TransformFactory"]:
        return [self]

    def get_required_inputs(self, requested_names: Set[str]) -> Set[str]:
        required_names = set()
        intermediate_names = set()

        for name in sorted(requested_names):
            req, interm = _get_dependencies(name, self._factory_list)
            required_names |= req
            intermediate_names |= interm

            overlap = required_names & intermediate_names

            if overlap:
                raise ValueError(
                    "The following variables in the transform chain appeared in"
                    f"the required inputs and intermediate variables: {overlap}"
                    f"  Adjust transform order such that creation of {overlap} precedes"
                    f" the creation of {name}"
                )

        return required_names


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

    @property
    def required_names(self) -> Set[str]:
        return {self.source}


@dataclasses.dataclass
class Difference(TransformFactory):

    to: str
    before: str
    after: str

    def backward_names(self, requested_names: Set[str]) -> Set[str]:
        new_names = {self.before, self.after} if self.to in requested_names else set()
        return requested_names.union(new_names)

    @property
    def required_names(self) -> Set[str]:
        return {self.before, self.after}

    def build(self, sample: TensorDict) -> TensorTransform:
        return DifferenceTransform(self.to, self.before, self.after)


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

    @property
    def required_names(self) -> Set[str]:
        return {self.condition_on, self.source}

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
        self.to = ""

    def backward_names(self, requested_names: Set[str]) -> Set[str]:
        for factory in self.factories:
            requested_names = factory.backward_names(requested_names)
        return requested_names

    def build(self, sample: TensorDict) -> ComposedTransform:
        transforms = [factory.build(sample) for factory in self.factories]
        return ComposedTransform(transforms)

    @property
    def _factory_list(self) -> List[TransformFactory]:

        expanded_list: List[TransformFactory] = []
        for f in self.factories:
            expanded_list += f._factory_list
        return expanded_list
