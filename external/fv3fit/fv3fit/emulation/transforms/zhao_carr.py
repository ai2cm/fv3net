"""zhao carr specific transformations

These will typically depend on the variable names used by the zhao carr
microphysics
"""
import dataclasses
from typing import Set, List, Tuple

import tensorflow as tf
from fv3fit.emulation.types import TensorDict
from .transforms import Difference, TensorTransform, LogTransform
from .factories import (
    ComposedTransformFactory,
    ConditionallyScaled,
    TransformedVariableConfig,
    TransformFactory,
)

# from physcons.f
latent_heat = 2.5e6
specific_heat = 1.0046e3

POSITIVE_TENDENCY = "positive_tendency"
ZERO_TENDENCY = "zero_tendency"
ZERO_CLOUD = "zero_cloud"
NEGATIVE_TENDENCY = "negative_tendency"
NONTRIVIAL_TENDENCY = "nontrivial_tendency"

# this constant is reused elswhere so is effectively public api
CLASS_NAMES = {
    POSITIVE_TENDENCY,
    ZERO_TENDENCY,
    ZERO_CLOUD,
    NEGATIVE_TENDENCY,
}

DELP = "pressure_thickness_of_atmospheric_layer"
PRESSURE = "air_pressure"
SURFACE_PRESSURE = "surface_air_pressure"
CLOUD_INPUT = "cloud_water_mixing_ratio_input"
CLOUD_GSCOND = "cloud_water_mixing_ratio_after_gscond"
CLOUD_PRECPD = "cloud_water_mixing_ratio_after_precpd"
T_LAST = "air_temperature_after_last_gscond"
T_INPUT = "air_temperature_input"
T_GSCOND = "air_temperature_after_gscond"
T_PRECPD = "air_temperature_after_precpd"
QV_LAST = "specific_humidity_after_last_gscond"
QV_INPUT = "specific_humidity_input"
QV_GSCOND = "specific_humidity_after_gscond"
QV_PRECPD = "specific_humidity_after_precpd"


def limit_negative_cloud(
    *, cloud: tf.Tensor, humidity: tf.Tensor, temperature: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    condensation = tf.where(cloud < 0, -cloud, 0.0)
    # don not make negative humidity
    condensation = tf.minimum(condensation, humidity)
    humidity_out = humidity - condensation
    temperature_out = temperature + condensation * latent_heat / specific_heat
    return cloud + condensation, humidity_out, temperature_out


@dataclasses.dataclass
class CloudLimiter(TensorTransform):
    """
    A hardcoded classification transform to assess cloud state/tendency
    behavior
    """

    cloud: str
    humidity: str
    temperature: str

    def build(self, sample: TensorDict) -> TensorTransform:
        return self

    def backward_input_names(self) -> Set[str]:
        return {
            self.cloud,
            self.humidity,
            self.temperature,
        }

    def backward_output_names(self) -> Set[str]:
        return {self.cloud, self.humidity, self.temperature}

    def backward_names(self, requested_names: Set[str]) -> Set[str]:
        return requested_names

    def forward(self, x: TensorDict) -> TensorDict:
        return x

    def backward(self, y: TensorDict) -> TensorDict:
        out = {**y}
        (
            out[self.cloud],
            out[self.humidity],
            out[self.temperature],
        ) = limit_negative_cloud(
            cloud=y[self.cloud],
            humidity=y[self.humidity],
            temperature=y[self.temperature],
        )
        return out


@dataclasses.dataclass
class MicrophysicsClasssesV1(TensorTransform):
    """
    A hardcoded classification transform to assess cloud state/tendency
    behavior
    """

    cloud_in: str = "cloud_water_mixing_ratio_input"
    cloud_out: str = "cloud_water_mixing_ratio_after_gscond"
    timestep: int = 900

    def build(self, sample: TensorDict) -> TensorTransform:
        return self

    def backward_names(self, requested_names: Set[str]) -> Set[str]:

        requested_names = set(requested_names)

        requested_names -= CLASS_NAMES
        requested_names -= {NONTRIVIAL_TENDENCY}
        requested_names |= {
            self.cloud_in,
            self.cloud_out,
        }
        return requested_names

    def forward(self, x: TensorDict) -> TensorDict:
        x = {**x}
        classes = classify(x[self.cloud_in], x[self.cloud_out], self.timestep)
        classes[NONTRIVIAL_TENDENCY] = (
            classes[POSITIVE_TENDENCY] | classes[NEGATIVE_TENDENCY]
        )
        x.update(classes)
        return x

    def backward(self, y: TensorDict) -> TensorDict:
        return y


@dataclasses.dataclass
class MicrophysicsClassesV1OneHot(MicrophysicsClasssesV1):
    to: str = "gscond_classes"

    def build(self, sample: TensorDict):
        # ensure classes are always encoded in the same order
        classes = classify(sample[self.cloud_in], sample[self.cloud_out], self.timestep)
        self._names = sorted(classes)
        return self

    def backward_names(self, requested_names: Set[str]) -> Set[str]:

        requested_names = set(requested_names)

        if self.to in requested_names:
            requested_names -= {self.to}
            requested_names |= {
                self.cloud_in,
                self.cloud_out,
            }
        return requested_names

    def forward(self, x: TensorDict) -> TensorDict:
        x = {**x}
        # python dicts remember order in 3.7 and later
        classes = classify(x[self.cloud_in], x[self.cloud_out], self.timestep)
        x[self.to] = tf.stack([classes[name] for name in self._names], -1)
        return x


def _combine(
    cloud_before, t_before, t_after, qv_before, qv_after, zero_tendency, zero_cloud,
):
    # compute net condensation = Condensation - Evap
    condensation = qv_before - qv_after
    cloud_after = cloud_before + condensation

    # apply no change case
    cloud_out = tf.where(zero_tendency, cloud_before, cloud_after)
    t_out = tf.where(zero_tendency, t_before, t_after)
    q_out = tf.where(zero_tendency, qv_before, qv_after)

    # apply zero cloud case
    # TODO fix these constants
    cloud_out = tf.where(zero_cloud, 0.0, cloud_out)
    q_out = tf.where(zero_cloud, qv_before + cloud_before, q_out)
    t_out = tf.where(
        zero_cloud, t_before - cloud_before * latent_heat / specific_heat, t_out
    )

    return cloud_out, t_out, q_out


@dataclasses.dataclass
class GscondRoute(TensorTransform):
    gscond_route: bool = True

    def build(self, sample: TensorDict):
        return self

    def backward_names(self, requested_names: Set[str]) -> Set[str]:
        return requested_names

    def backward_input_names(self) -> Set[str]:
        return {
            CLOUD_INPUT,
            T_INPUT,
            T_GSCOND,
            QV_INPUT,
            QV_GSCOND,
            ZERO_TENDENCY,
            ZERO_CLOUD,
        }

    def backward_output_names(self) -> Set[str]:
        return {CLOUD_GSCOND}

    def forward(self, x: TensorDict) -> TensorDict:
        return x

    def backward(self, y: TensorDict) -> TensorDict:
        y = {**y}
        cloud_out, t_out, qv_out = _combine(
            cloud_before=y[CLOUD_INPUT],
            t_before=y[T_INPUT],
            t_after=y[T_GSCOND],
            qv_before=y[QV_INPUT],
            qv_after=y[QV_GSCOND],
            zero_tendency=y[ZERO_TENDENCY],
            zero_cloud=y[ZERO_CLOUD],
        )
        y[CLOUD_GSCOND] = cloud_out
        y[T_GSCOND] = t_out
        y[QV_GSCOND] = qv_out
        return y


def classify(cloud_in, cloud_out, timestep, math=tf.math):
    state_thresh = 1e-15
    tend_thresh = 1e-15

    tend = (cloud_out - cloud_in) / timestep
    some_cloud_out = math.abs(cloud_out) > state_thresh
    negative_tend = tend < -tend_thresh

    return {
        POSITIVE_TENDENCY: tend > tend_thresh,
        ZERO_TENDENCY: math.abs(tend) <= tend_thresh,
        ZERO_CLOUD: negative_tend & ~some_cloud_out,
        NEGATIVE_TENDENCY: negative_tend & some_cloud_out,
    }


class GscondOnly(TransformFactory):
    """A python Transform Factory encoding this configuration


    tensor_transform:
    - to: log_cloud_input
        source: cloud_water_mixing_ratio_input
        transform: {epsilon: 1e-10}
    - to: log_humidity_input
        source: specific_humidity_input
        transform: {epsilon: 1e-8}
    - to: log_humidity_after_last_gscond
        source: specific_humidity_after_last_gscond
        transform: {epsilon: 1e-8}
    - to: temperature_gscond_difference
        before: air_temperature_input
        after: air_temperature_after_gscond
    - to: humidity_gscond_difference
        before: specific_humidity_input
        after: specific_humidity_after_gscond
    - to: humidity_gscond_difference_tscaled
        source: humidity_gscond_difference
        condition_on: air_temperature_input
        bins: 50
        min_scale: 1e-14
        fit_filter_magnitude: 1e-14
    - to: temperature_gscond_difference_tscaled
        source: temperature_gscond_difference
        condition_on: air_temperature_input
        bins: 50
        min_scale: 1e-5
        fit_filter_magnitude: 1e-5
    """

    def backward_names(self, requested_names: Set[str]) -> Set[str]:
        return self._composed().backward_names(requested_names)

    def _composed(self) -> TransformFactory:
        t_diff = "temperature_gscond_difference"
        t_diff_scale = "temperature_gscond_difference_tscaled"
        qv_diff = "humidity_gscond_difference"
        qv_diff_scale = "humidity_gscond_difference_tscaled"
        factories: List[TransformFactory] = [
            TransformedVariableConfig(
                CLOUD_INPUT, to="log_cloud_input", transform=LogTransform(1e-10)
            ),
            TransformedVariableConfig(
                QV_INPUT, to="log_humidity_input", transform=LogTransform(1e-8)
            ),
            TransformedVariableConfig(
                QV_LAST,
                to="log_humidity_after_last_gscond",
                transform=LogTransform(1e-8),
            ),
            Difference(to=t_diff, before=T_INPUT, after=T_GSCOND),
            Difference(to=qv_diff, before=QV_INPUT, after=QV_GSCOND),
            ConditionallyScaled(
                to=t_diff_scale,
                condition_on=T_INPUT,
                source=t_diff,
                bins=50,
                min_scale=1e-5,
                fit_filter_magnitude=1e-5,
            ),
            ConditionallyScaled(
                to=qv_diff_scale,
                condition_on=T_INPUT,
                source=qv_diff,
                bins=50,
                min_scale=1e-14,
                fit_filter_magnitude=1e-14,
            ),
        ]
        return ComposedTransformFactory(factories)

    def build(self, sample: TensorDict) -> TensorTransform:
        return self._composed().build(sample)


@dataclasses.dataclass
class PrecpdOnly(TransformFactory):
    """A Transform Factory for precpd only prediction with inputs as output

    """

    t_diff = "temperature_precpd_difference"
    t_diff_scale = "temperature_precpd_difference_tscaled"
    qv_diff = "humidity_precpd_difference"
    qv_diff_scale = "humidity_precpd_difference_tscaled"
    qc_diff = "cloud_precpd_difference"
    qc_diff_scale = "cloud_precpd_difference_tscaled"

    log_cloud_input = "log_cloud_input"
    log_humidity_input = "log_humidity_input"
    # useless field to disambiguate with
    precpd_only: bool = True

    def backward_names(self, requested_names: Set[str]) -> Set[str]:
        return self._composed().backward_names(requested_names)

    def _composed(self) -> TransformFactory:
        factories: List[TransformFactory] = [
            TransformedVariableConfig(
                CLOUD_GSCOND, to=self.log_cloud_input, transform=LogTransform(1e-10)
            ),
            TransformedVariableConfig(
                QV_GSCOND, to=self.log_humidity_input, transform=LogTransform(1e-8)
            ),
            Difference(to=self.t_diff, before=T_GSCOND, after=T_PRECPD),
            Difference(to=self.qv_diff, before=QV_GSCOND, after=QV_PRECPD),
            Difference(to=self.qc_diff, before=CLOUD_GSCOND, after=CLOUD_PRECPD),
            ConditionallyScaled(
                to=self.t_diff_scale,
                condition_on=T_GSCOND,
                source=self.t_diff,
                bins=50,
                min_scale=1e-5,
                fit_filter_magnitude=1e-5,
            ),
            ConditionallyScaled(
                to=self.qv_diff_scale,
                condition_on=T_GSCOND,
                source=self.qv_diff,
                bins=50,
                min_scale=1e-14,
                fit_filter_magnitude=1e-14,
            ),
            ConditionallyScaled(
                to=self.qc_diff_scale,
                condition_on=T_GSCOND,
                source=self.qc_diff,
                bins=50,
                min_scale=1e-14,
                fit_filter_magnitude=1e-14,
            ),
        ]
        return ComposedTransformFactory(factories)

    def build(self, sample: TensorDict) -> TensorTransform:
        return self._composed().build(sample)
