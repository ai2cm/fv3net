import numpy as np


def get_latitude(state):
    return state["latitude"]


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
