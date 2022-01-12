import numpy as np

_qc_out = "cloud_water_mixing_ratio_after_precpd"


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
