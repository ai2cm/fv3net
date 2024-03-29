import dataclasses
from typing import Callable, List, Union, Set, Optional

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

    Notes:
        This class is its own factory (i.e. includes the .build and
        .backwards_names methods). This is only possible because it doesn't
        depend on data and can be represented directly in yaml.

    """

    to: str
    before: str
    after: str

    def backward_input_names(self) -> Set[str]:
        return {self.to, self.before}

    def backward_output_names(self) -> Set[str]:
        return {self.after}

    def backward_names(self, requested_names: Set[str]) -> Set[str]:

        if self.to in requested_names:
            requested_names = (requested_names - {self.to}) | {self.before, self.after}

        return requested_names

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
class CloudWaterDiffPrecpd(TensorTransform):
    """
    A specialized transform for patching in some necessary information
    about cloud water after gscond in order to provide information on the
    cloud tendencies from the Zhao-Carr precpd scheme.

    Once `cloud_water_mixing_ratio_after_gscond` is in the dataset, this
    transform will no longer be needed.
    """

    to: str
    # input names
    sphum_source: str
    cloud_input: str
    cloud_after_precpd: str

    def build(self, sample: TensorDict) -> TensorTransform:
        return self

    def backward_names(self, requested_names: Set[str]) -> Set[str]:

        if self.to in requested_names:
            requested_names -= {self.to}
            requested_names |= {
                self.sphum_source,
                self.cloud_input,
                self.cloud_after_precpd,
            }
        return requested_names

    def _get_cloud_after_gscond(self, x: TensorDict) -> tf.Tensor:
        cloud_source = tf.constant(-1.0, dtype=tf.float32) * x[self.sphum_source]
        cloud_after_gscond = x[self.cloud_input] + cloud_source
        return cloud_after_gscond

    def forward(self, x: TensorDict):
        x = {**x}
        cloud_after_gscond = self._get_cloud_after_gscond(x)
        x[self.to] = x[self.cloud_after_precpd] - cloud_after_gscond
        return x

    def backward(self, y: TensorDict):
        y = {**y}
        cloud_after_gscond = self._get_cloud_after_gscond(y)
        y[self.cloud_after_precpd] = cloud_after_gscond + y[self.to]
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


@dataclasses.dataclass
class LimitValueTransform:
    """
    A univariate transformation for::

    y := x
    x := y where lower_limit < y < upper limit, 0 elsewhere

    Attributes:
        lower: lower bound for value clipping
        upper: upper bound for value clipping
    """

    lower: Optional[float] = 0.0
    upper: Optional[float] = None

    def forward(self, x: tf.Tensor) -> tf.Tensor:
        return x

    def backward(self, x: tf.Tensor) -> tf.Tensor:

        if self.lower is not None:
            x = tf.keras.activations.relu(x, threshold=self.lower)

        if self.upper is not None:
            x = tf.cast(x < self.upper, x.dtype) * x

        return x


UnivariateCompatible = Union[
    LogTransform, LimitValueTransform,
]


class UnivariateTransform(TensorTransform):
    def __init__(
        self, source: str, transform: UnivariateCompatible, to: Optional[str] = None
    ):
        self.source = source
        self.to = to or source
        self.transform = transform

    def forward(self, x: TensorDict) -> TensorDict:
        out = {**x}
        out[self.to] = self.transform.forward(x[self.source])
        return out

    def backward(self, y: TensorDict) -> TensorDict:
        out = {**y}
        out[self.source] = self.transform.backward(y[self.to])
        return out

    def backward_input_names(self) -> Set[str]:
        return {self.to}

    def backward_output_names(self) -> Set[str]:
        return {self.source}


class ConditionallyScaledTransform(TensorTransform):
    def __init__(
        self,
        to: str,
        source: str,
        on: str,
        scale: Callable[[tf.Tensor], tf.Tensor],
        center: Callable[[tf.Tensor], tf.Tensor],
        min_scale: float = 0.0,
    ) -> None:
        self.to = to
        self.source = source
        self.on = on
        self.scale = scale
        self.center = center
        self.min_scale = min_scale

    def _limited_scale(self, x: tf.Tensor) -> tf.Tensor:
        return tf.maximum(self.scale(x), self.min_scale)

    def forward(self, x: TensorDict) -> TensorDict:
        out = {**x}
        out[self.to] = (x[self.source] - self.center(x[self.on])) / self._limited_scale(
            x[self.on]
        )
        return out

    def backward(self, y: TensorDict) -> TensorDict:
        out = {**y}
        out[self.source] = y[self.to] * self._limited_scale(y[self.on]) + self.center(
            y[self.on]
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
