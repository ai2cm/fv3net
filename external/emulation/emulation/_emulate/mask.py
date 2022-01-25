import numpy as np
from dataclasses import dataclass
import scipy.interpolate
from typing import Optional
from . import mask_data
from emulation._typing import FortranState

_qc_out = "cloud_water_mixing_ratio_after_precpd"
_temp_out = "air_temperature_after_precpd"

max_cloud_from_temp = scipy.interpolate.interp1d(
    mask_data.temp, mask_data.qc, fill_value=1e-7, bounds_error=False
)


def get_temperature(state):
    return state[_temp_out]


def get_latitude(state):
    return state["latitude"]


def get_cloud_output(state):
    return state[_qc_out]


def assoc_cloud_output(state, val):
    out = state.copy()
    out[_qc_out] = val
    return out


def is_outside_lat_range(state, lat_range=(-60, 60)):
    """return true where lat is < 60"""
    latitude = get_latitude(state)
    min, max = lat_range

    return (latitude < min) | (latitude > max)


def where(mask, left_state, right_state):
    """Where mask is True use left state"""
    out = {}
    n, one = mask.shape
    assert one == 1, mask.shape
    assert set(left_state) == set(right_state)

    for key in left_state:
        left, right = left_state[key], right_state[key]
        assert n == left.shape[0], key
        assert n == right.shape[0], key
        out[key] = np.where(mask, left, right)

    return out


def threshold_clouds(state, max):
    qc = get_cloud_output(state)
    qc_thresh = np.where(qc > max, max, qc)
    return assoc_cloud_output(state, qc_thresh)


def threshold_clouds_temperature_dependent(state):
    """threshold cloud outputs with temperature dependent threshold"""
    t = get_temperature(state)
    qc = get_cloud_output(state)

    max_qc = max_cloud_from_temp(t)
    qc_thresh = np.where(qc > max_qc, max_qc, qc)
    return assoc_cloud_output(state, qc_thresh)


@dataclass
class MaskConfig:
    max_cloud: Optional[float] = None
    temperature_dependent_max: bool = False
    max_lat: Optional[float] = None
    min_lat: Optional[float] = None

    def __call__(
        self, inputs: FortranState, outputs: FortranState, predictions: FortranState,
    ) -> FortranState:
        if self.max_lat or self.min_lat:
            lat_range = (self.min_lat or -100, self.max_lat or 100)
            lat_mask = is_outside_lat_range(inputs, lat_range=lat_range)
            predictions = where(lat_mask, outputs, predictions)

        if self.max_cloud:
            predictions = threshold_clouds(predictions, max=self.max_cloud)

        if self.temperature_dependent_max:
            predictions = threshold_clouds_temperature_dependent(predictions)

        return predictions
