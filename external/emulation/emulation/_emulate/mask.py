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

    for key in set(left_state).union(right_state):
        if key in left_state and key in right_state:
            left, right = left_state[key], right_state[key]
            assert n == left.shape[0], key
            assert n == right.shape[0], key
            out[key] = np.where(mask, left, right)
        elif key in left_state:
            out[key] = left_state[key]
        elif key in right_state:
            out[key] = right_state[key]

    return out
