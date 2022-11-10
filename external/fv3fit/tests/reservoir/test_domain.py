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
    domain = PeriodicDomain(np.arange(8), subdomain_size=2, subdomain_overlap=2)
    subdomain_start = domain[0]
    subdomain_end = domain[3]
    np.testing.assert_array_equal(
        subdomain_start.overlapping, np.array([6, 7, 0, 1, 2, 3])
    )
    np.testing.assert_array_equal(
        subdomain_end.overlapping, np.array([4, 5, 6, 7, 0, 1])
    )


def test_PeriodicDomain_iteration_elements():
    domain = PeriodicDomain(np.arange(6), subdomain_size=2, subdomain_overlap=1)
    expected_subdomains_without_overlap = [
        np.array([0, 1]),
        np.array([2, 3]),
        np.array([4, 5]),
    ]
    expected_subdomains_with_overlap = [
        np.array([5, 0, 1, 2]),
        np.array([1, 2, 3, 4]),
        np.array([3, 4, 5, 0]),
    ]
    for expected_no_overlap, expected_with_overlap, subdomain in zip(
        expected_subdomains_without_overlap, expected_subdomains_with_overlap, domain
    ):
        np.testing.assert_array_equal(expected_with_overlap, subdomain.overlapping)
        np.testing.assert_array_equal(expected_no_overlap, subdomain.nonoverlapping)


def test_PeriodicDomain_len():
    domain = PeriodicDomain(np.arange(6), subdomain_size=2, subdomain_overlap=1)
    assert len(domain) == 3


def test_PeriodicDomain_iteration():
    domain = PeriodicDomain(np.arange(6), subdomain_size=2, subdomain_overlap=1)
    count = 0
    for subdomain in domain:
        count += 1
    assert count == 3


def test_PeriodicDomain_raises_index_out_of_range():
    domain = PeriodicDomain(np.arange(6), subdomain_size=2, subdomain_overlap=1)
    with pytest.raises(ValueError):
        domain[5]
