import numpy as np
from .regrid import interpolate_1d_nd_target, lagrangian_origin_coordinates


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


def test_semi_lagrangian():
    nt, nx, ny, nz = 1, 3, 3, 3
    ndim = 4

    x = np.r_[:nx]
    y = np.r_[:ny]
    z = np.broadcast_to(np.r_[:nz].reshape((1, -1, 1, 1)), (nt, nz, ny, nx))

    u = np.ones_like(z)
    v = np.ones_like(z)
    w = np.ones_like(z)

    ans = lagrangian_origin_coordinates(x, y, z, u, v, w, h=1)

    assert ans.shape == (nt, nx, ny, nz, ndim), ans.shape

    test_point = ans[0, 2, 2, 2, :]
    assert test_point.tolist() == [0, 1, 1, 1]

    test_point = ans[0, 1, 1, 1, :]
    assert test_point.tolist() == [0, 0, 0, 0]

    test_point = ans[0, 0, 1, 1, :]
    assert test_point.tolist() == [0, -1, 0, 0]
