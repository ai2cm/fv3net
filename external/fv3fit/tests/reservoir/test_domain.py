import numpy as np
import pytest

from fv3fit.reservoir.domain import (
    RankDivider,
    stack_time_series_samples,
    slice_along_axis,
)


arr = np.arange(3)


@pytest.mark.parametrize(
    "data, ax, sl, expected",
    [
        (arr, 0, slice(1, None), np.array([1, 2])),
        (np.array([arr, arr, arr]), 0, slice(0, 2), np.array([arr, arr])),
        (np.array([arr, arr]), 1, slice(0, 2), np.array([[0, 1], [0, 1]])),
    ],
)
def test_slice_along_axis(data, ax, sl, expected):
    np.testing.assert_array_equal(slice_along_axis(data, axis=ax, inds=sl), expected)


@pytest.mark.parametrize(
    "layout, rank_extent, overlap, expected_with_overlap, expected_without_overlap",
    [
        [(2, 2), (5, 6, 6, 2), 0, (5, 3, 3, 2), (5, 3, 3, 2)],
        [(2, 2), (5, 6, 6, 2), 1, (5, 4, 4, 2), (5, 2, 2, 2)],
        [(2, 2), (5, 10, 10, 2), 2, (5, 7, 7, 2), (5, 3, 3, 2)],
    ],
)
def test_RankDivider_get_subdomain_extent(
    layout, rank_extent, overlap, expected_with_overlap, expected_without_overlap
):
    divider = RankDivider(
        subdomain_layout=layout,
        rank_dims=["time", "x", "y", "z"],
        rank_extent=rank_extent,
        overlap=overlap,
    )
    assert divider.get_subdomain_extent(with_overlap=True) == expected_with_overlap
    assert divider.get_subdomain_extent(with_overlap=False) == expected_without_overlap


def test_RankDivider_subdomain_slice():
    nt, nz = 5, 2
    divider = RankDivider(
        subdomain_layout=(2, 2),
        rank_dims=["time", "x", "y", "z"],
        rank_extent=[nt, 6, 6, nz],
        overlap=1,
    )
    time_slice = slice(0, nt)
    z_slice = slice(0, nz)

    assert divider.subdomain_slice(0, with_overlap=True) == (
        time_slice,
        slice(0, 4),
        slice(0, 4),
        z_slice,
    )
    assert divider.subdomain_slice(0, with_overlap=False) == (
        time_slice,
        slice(1, 3),
        slice(1, 3),
        z_slice,
    )
    assert divider.subdomain_slice(3, with_overlap=True) == (
        time_slice,
        slice(2, 6),
        slice(2, 6),
        z_slice,
    )
    assert divider.subdomain_slice(3, with_overlap=False) == (
        time_slice,
        slice(3, 5),
        slice(3, 5),
        z_slice,
    )


def test_RankDivider_subdomain_tensor_slice_overlap():
    xy_arr = np.pad(np.arange(1, 5).reshape(2, 2), pad_width=1)
    arr = np.reshape(xy_arr, (1, 4, 4, 1))
    divider = RankDivider(
        subdomain_layout=(2, 2),
        rank_dims=["time", "x", "y", "z"],
        rank_extent=[1, 4, 4, 1],
        overlap=1,
    )
    bottom_right_with_overlap = divider.get_subdomain_tensor_slice(
        arr, 3, with_overlap=True
    )
    np.testing.assert_array_equal(
        bottom_right_with_overlap[0, :, :, 0],
        np.array([[1, 2, 0], [3, 4, 0], [0, 0, 0]]),
    )

    bottom_right_no_overlap = divider.get_subdomain_tensor_slice(
        arr, 3, with_overlap=False
    )
    np.testing.assert_array_equal(bottom_right_no_overlap[0, :, :, 0], np.array([[4]]))


def test_RankDivider_get_subdomain_tensor_slice():
    divider = RankDivider(
        subdomain_layout=(2, 2),
        rank_dims=["time", "x", "y", "z"],
        rank_extent=[1, 2, 2, 1],
        overlap=0,
    )
    arr = np.arange(1, 5).reshape(1, 2, 2, 1)
    subdomain_values = []
    for s in range(divider.n_subdomains):
        subdomain_values.append(
            divider.get_subdomain_tensor_slice(arr, s, with_overlap=False)
            .flatten()
            .item()
        )

    assert set(subdomain_values) == {1, 2, 3, 4}


def test_stack_time_series_samples():
    time_series = np.array([np.ones((2, 2)) * i for i in range(10)])
    stacked = stack_time_series_samples(time_series)
    np.testing.assert_array_equal(stacked[-1], np.array([9, 9, 9, 9]))


def test_RankDivider_unstack_subdomain():
    nt, nx, ny, nz = 5, 6, 6, 2
    divider = RankDivider(
        subdomain_layout=(2, 2),
        rank_dims=["time", "x", "y", "z"],
        rank_extent=[nt, nx, ny, nz],
        overlap=1,
    )
    rank_arr = np.random.randn(nt, nx, ny, nz)
    subdomain_arr = divider.get_subdomain_tensor_slice(rank_arr, 0, with_overlap=True)
    stacked = stack_time_series_samples(subdomain_arr)
    np.testing.assert_array_equal(
        divider.unstack_subdomain(stacked, with_overlap=True), subdomain_arr
    )
