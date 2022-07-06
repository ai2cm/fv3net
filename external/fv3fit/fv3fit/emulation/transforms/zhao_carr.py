import dataclasses
from typing import Set

import tensorflow as tf
from fv3fit.emulation.types import TensorDict
from .transforms import TensorTransform

POSITIVE_TENDENCY = "positive_tendency"
ZERO_TENDENCY = "zero_tendency"
ZERO_CLOUD = "zero_cloud"
NEGATIVE_TENDENCY = "negative_tendency"

CLASS_NAMES = {
    POSITIVE_TENDENCY,
    ZERO_TENDENCY,
    ZERO_CLOUD,
    NEGATIVE_TENDENCY,
}


@dataclasses.dataclass
class GscondClassesV1(TensorTransform):
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

        if CLASS_NAMES & requested_names:
            requested_names -= CLASS_NAMES
            requested_names |= {
                self.cloud_in,
                self.cloud_out,
            }
        return requested_names

    def forward(self, x: TensorDict) -> TensorDict:
        x = {**x}
        classes = classify(x[self.cloud_in], x[self.cloud_out], self.timestep)
        x.update(classes)
        return x

    def backward(self, y: TensorDict) -> TensorDict:
        return y


@dataclasses.dataclass
class GscondClassesV1OneHot(GscondClassesV1):
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


# TODO: Probably a V2 of this class that just uses gscond outputs
@dataclasses.dataclass
class PrecpdClassesV1(GscondClassesV1):
    cloud_in: str = "cloud_water_mixing_ratio_input"
    cloud_out: str = "cloud_water_mixing_ratio_after_precpd"
    timestep: int = 900


@dataclasses.dataclass
class PrecpdClassesV1OneHot(GscondClassesV1OneHot):
    cloud_in: str = "cloud_water_mixing_ratio_input"
    cloud_out: str = "cloud_water_mixing_ratio_after_precpd"
    timestep: int = 900
    to: str = "precpd_classes"


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
