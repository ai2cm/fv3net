import numpy as np
from vcm.advect import (
    interpolate_1d_nd_target,
    lagrangian_origin_coordinates,
    compute_dz,
)


def test_interpolate_1d_nd_target_1d_input():
    x = np.array([0, 1])
    y = np.array([0, 2])

    new_grid = np.array([0.5, 0.75])
    expected = np.array([1.0, 1.5])
    answer = interpolate_1d_nd_target(new_grid, x, y)
    np.testing.assert_allclose(expected, answer)


def test_interpolate_1d_nd_target_nd_input():
    x = np.array([0, 1])
    y = np.array([[0, 1], [0, 1]])
    new_grid = np.array([[0.5, 0.75], [0, 1]])

    expected = np.array([[0.5, 0.75], [0, 1]])
    answer = interpolate_1d_nd_target(new_grid, x, y)
    np.testing.assert_allclose(expected, answer)


def test_semi_lagrangian():
    nt, nx, ny, nz = 1, 3, 3, 3
    ndim = 4

    x = np.r_[:nx]
    y = np.r_[:ny]

    x, y = np.meshgrid(x, y)
    dx = np.ones((ny, nx))
    dy = np.ones((ny, nx))
    dz = np.ones((nt, nx, ny, nz))

    u = np.ones_like(dz)
    v = np.ones_like(dz)
    w = np.ones_like(dz)

    ans = lagrangian_origin_coordinates(dx, dy, dz, u, v, w, h=1)

    assert ans.shape == (ndim, nt, nx, ny, nz), ans.shape

    test_point = ans[:, 0, 2, 2, 2]
    assert test_point.tolist() == [0, 1, 1, 1]

    test_point = ans[:, 0, 1, 1, 1]
    assert test_point.tolist() == [0, 0, 0, 0]

    test_point = ans[:, 0, 0, 1, 1]
    assert test_point.tolist() == [0, -1, 0, 0]


def test_compute_dz():
    spacing = 2
    shape = (1, 3, 2, 2)
    expected = np.ones(shape) * spacing

    nz = shape[1]
    z = np.arange(nz).reshape((1, -1, 1, 1)) * spacing
    z = np.broadcast_to(z, shape)
    dz = compute_dz(z)
    np.testing.assert_almost_equal(dz, expected)
