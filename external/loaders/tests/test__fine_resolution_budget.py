import numpy as np
from loaders.mappers._hybrid import _convergence


def test__convergence_constant():
    nz = 5
    delp = np.ones(nz).reshape((1, 1, nz))

    expected = np.array([0, 0, 0, 0, 0]).reshape((1, 1, nz))

    ans = _convergence(delp, delp)
    np.testing.assert_almost_equal(ans, expected)


def test__convergence_linear():
    nz = 5
    f = np.arange(nz).reshape((1, 1, nz))
    delp = np.ones(nz).reshape((1, 1, nz))

    expected = np.array([-1, -1, -1, -1, -1]).reshape((1, 1, nz))

    ans = _convergence(f, delp)
    np.testing.assert_almost_equal(ans, expected)
