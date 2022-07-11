"""zhao carr specific transformations

These will typically depend on the variable names used by the zhao carr
microphysics
"""
import dataclasses
from typing import Set

import tensorflow as tf
from fv3fit.emulation.types import TensorDict
from .transforms import TensorTransform

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

CLOUD_INPUT = "cloud_water_mixing_ratio_input"
CLOUD_GSCOND = "cloud_water_mixing_ratio_after_gscond"
T_INPUT = "air_temperature_input"
T_GSCOND = "air_temperature_after_gscond"
QV_INPUT = "specific_humidity_input"
QV_GSCOND = "specific_humidity_after_gscond"


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
    latent_heat = 2.51e6
    specific_heat = 1004
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
