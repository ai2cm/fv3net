import numpy as np
import pytest
from fv3fit.reservoir.domain import PeriodicDomain, _slice


arr = np.arange(3)


@pytest.mark.parametrize(
    "data, ax, sl, expected",
    [
        (arr, 0, slice(1, None), np.array([1, 2])),
        (np.array([arr, arr, arr]), 0, slice(0, 2), np.array([arr, arr])),
        (np.array([arr, arr]), 1, slice(0, 2), np.array([[0, 1], [0, 1]])),
    ],
)
def test__slice(data, ax, sl, expected):
    np.testing.assert_array_equal(_slice(data, axis=ax, inds=sl), expected)


def test_PeriodicDomain():
    domain = PeriodicDomain(np.arange(8), output_size=2, overlap=2)
    subdomain_start = domain[0]
    subdomain_end = domain[3]
    np.testing.assert_array_equal(
        subdomain_start.overlapping, np.array([6, 7, 0, 1, 2, 3])
    )
    np.testing.assert_array_equal(
        subdomain_end.overlapping, np.array([4, 5, 6, 7, 0, 1])
    )
