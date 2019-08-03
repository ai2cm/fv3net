import numpy as np
from .regrid import interpolate_1d_nd_target


def test_interpolate_1d_nd_target_1d_input():
    x = np.array([0, 1])
    y = np.array([0, 2])

    new_grid = np.array([.5, .75])
    expected = np.array([1.0, 1.5])
    answer = interpolate_1d_nd_target(new_grid, x, y)
    np.testing.assert_allclose(expected, answer)


def test_interpolate_1d_nd_target_nd_input():
    x = np.array([0, 1])
    y = np.array([[0, 1], [0, 1]])
    new_grid = np.array([[.5, .75], [0, 1]])

    expected = np.array([[.5, .75], [0, 1]])
    answer = interpolate_1d_nd_target(new_grid, x, y)
    np.testing.assert_allclose(expected, answer)
