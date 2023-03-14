import numpy as np
import pytest
from fv3fit.reservoir.domain import (
    slice_along_axis,
    RankDivider,
    DataReshaper,
    concat_variables_along_feature_dim,
    stack_time_series_samples,
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


default_rank_divider_kwargs = {
    "layout": [2, 2],
    "overlap": 1,
    "rank_dims": ["time", "x", "y", "z"],
    "rank_extent": [5, 6, 6, 7],
}


def get_reshaper(variables, **rank_divider_kwargs):
    return DataReshaper(
        variables=variables, rank_divider=RankDivider(**rank_divider_kwargs)
    )


def test_concat_variables_along_feature_dim():
    nt, nx, ny, nz = 5, 4, 4, 6
    arr0 = np.zeros((nt, nx, ny, nz))
    arr1 = np.ones((nt, nx, ny, nz))
    data_mapping = {"var1": arr1, "var0": arr0}

    concat_data = concat_variables_along_feature_dim(["var0", "var1"], data_mapping)
    assert concat_data.shape == (nt, nx, ny, nz * 2)
    np.testing.assert_array_equal(concat_data[:, :, :, :nz], arr0)
    np.testing.assert_array_equal(concat_data[:, :, :, nz:], arr1)


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


@pytest.mark.parametrize(
    "rank_extent, overlap, with_overlap",
    [
        # ([5, 6, 6, 1], 1, True),
        # ([5, 6, 6, 2], 1, True),
        # ([5, 6, 6, 2], 1, False),
        # ([5, 6, 6, 2], 0, True),
        ([1, 4, 4, 3], 0, False),
    ],
)
def test_RankDivider_unstack_subdomain(rank_extent, overlap, with_overlap):
    divider = RankDivider(
        subdomain_layout=(2, 2),
        rank_dims=["time", "x", "y", "z"],
        rank_extent=rank_extent,
        overlap=overlap,
    )
    rank_arr = np.arange(np.prod(rank_extent)).reshape(*rank_extent)
    subdomain_arr = divider.get_subdomain_tensor_slice(
        rank_arr, 0, with_overlap=with_overlap
    )
    stacked = stack_time_series_samples(subdomain_arr)
    assert len(stacked.shape) == 2
    np.testing.assert_array_equal(
        divider.unstack_subdomain(stacked, with_overlap=with_overlap), subdomain_arr
    )


def test_RankDivider_flatten_subdomains_to_columns():
    nt, nx, ny, nz = 5, 6, 6, 2
    divider = RankDivider(
        subdomain_layout=(2, 2),
        rank_dims=["time", "x", "y", "z"],
        rank_extent=[nt, nx, ny, nz],
        overlap=1,
    )
    input = np.array(
        [[0, 1, 10, 11], [2, 3, 12, 13], [20, 21, 30, 31], [22, 23, 32, 33]]
    )
    input_with_halo = np.pad(input, pad_width=1)
    # add time dim of length 2
    input_with_halo = np.stack([input_with_halo, input_with_halo], axis=0)
    flattened = divider.flatten_subdomains_to_columns(
        input_with_halo, with_overlap=True
    )

    # 4 subdomains each with x, y dims (4, 4)
    assert flattened.shape == (2, 16, 4)
    # Check first subdomain of first sample has correct values
    np.testing.assert_array_almost_equal(
        flattened[0, :, 0],
        np.array([0, 0, 0, 0, 0, 0, 1, 10, 0, 2, 3, 12, 0, 20, 21, 30]),
    )


@pytest.mark.parametrize(
    "rank_extent, layout, overlap, expected_extent",
    [
        ([2, 8, 8, 5], [2, 2], 1, 3),
        ([1, 8, 8, 2], [2, 2], 1, 3),
        ([2, 8, 8, 5], [2, 2], 0, 4),
        ([1, 8, 8, 2], [2, 2], 0, 4),
    ],
)
def test_RankDivider__rank_extent_without_overlap(
    rank_extent, layout, overlap, expected_extent
):
    divider = RankDivider(
        subdomain_layout=layout,
        rank_dims=["time", "x", "y", "z"],
        rank_extent=rank_extent,
        overlap=overlap,
    )

    assert divider._rank_extent_without_overlap == expected_extent


def test_RankDivider_xy_size_before_overlap():
    nt, nx, ny, nz = 1, 8, 8, 2
    divider = RankDivider(
        subdomain_layout=(2, 2),
        rank_dims=["time", "x", "y", "z"],
        rank_extent=[nt, nx, ny, nz],
        overlap=2,
    )
    assert divider.xy_size_without_overlap == 2


def test_RankDivider_reshape_1d_to_2d_domain():
    rank_extent = [1, 8, 8, 3]
    divider = RankDivider(
        subdomain_layout=(2, 2),
        rank_dims=["time", "x", "y", "z"],
        rank_extent=rank_extent,
        overlap=2,
    )
    flattened = np.concatenate(
        [[(100 * i) + np.arange(12)] for i in range(4)], axis=0
    ).flatten()

    domain = np.array(
        [[0, 1, 10, 11], [2, 3, 12, 13], [20, 21, 30, 31], [22, 23, 32, 33]]
    )
    np.testing.assert_array_equal(
        divider.reshape_1d_to_2d_domain(flattened)[:, :, 0], domain
    )
