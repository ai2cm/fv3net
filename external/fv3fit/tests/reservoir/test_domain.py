import numpy as np
import pytest
import xarray as xr
from fv3fit.reservoir.domain import (
    slice_along_axis,
    RankDivider,
    TimeSeriesRankDivider,
    assure_same_dims,
    _transpose_xy_dims,
)
from fv3fit.reservoir._reshaping import stack_data


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
    "rank_dims": ["x", "y"],
    "rank_extent": [6, 6],
}


def test_assure_same_dims():
    nt, nx, ny, nz = 5, 4, 4, 6
    arr_3d = np.ones((nt, nx, ny, nz))
    arr_2d = np.ones((nt, nx, ny))
    data = [arr_3d, arr_2d]
    assert assure_same_dims(data)[0].shape == (nt, nx, ny, nz)
    assert assure_same_dims(data)[1].shape == (nt, nx, ny, 1)


def test_assure_same_dims_incompatible_shapes():
    nt, nx, ny, nz = 5, 4, 4, 6
    arr_3d = np.ones((nt, nx, ny, nz, 2))
    arr_2d = np.ones((nt, nx, ny))
    data = [arr_3d, arr_2d]
    with pytest.raises(ValueError):
        assure_same_dims(data)


@pytest.mark.parametrize(
    "layout, rank_dims, rank_extent, \
        overlap, expected_with_overlap, expected_without_overlap",
    [
        [(2, 2), ["x", "y"], (6, 6), 0, (3, 3), (3, 3)],
        [(2, 2), ["x", "y"], (6, 6), 1, (4, 4), (2, 2)],
    ],
)
def test_RankDivider_get_subdomain_extent(
    layout,
    rank_dims,
    rank_extent,
    overlap,
    expected_with_overlap,
    expected_without_overlap,
):
    divider = RankDivider(
        subdomain_layout=layout,
        rank_dims=rank_dims,
        rank_extent=rank_extent,
        overlap=overlap,
    )
    assert divider.get_subdomain_extent(with_overlap=True) == expected_with_overlap
    assert divider.get_subdomain_extent(with_overlap=False) == expected_without_overlap


def test_RankDivider_subdomain_slice():
    divider = RankDivider(
        subdomain_layout=(2, 2), rank_dims=["x", "y"], rank_extent=[6, 6], overlap=1,
    )

    assert divider.subdomain_slice(0, with_overlap=True) == (slice(0, 4), slice(0, 4))
    assert divider.subdomain_slice(0, with_overlap=False) == (slice(1, 3), slice(1, 3))
    assert divider.subdomain_slice(3, with_overlap=True) == (slice(2, 6), slice(2, 6))
    assert divider.subdomain_slice(3, with_overlap=False) == (slice(3, 5), slice(3, 5))


def test_TimeSeriesRankDivider_subdomain_tensor_slice_overlap():
    xy_arr = np.pad(np.arange(1, 5).reshape(2, 2), pad_width=1)
    arr = np.reshape(xy_arr, (1, 4, 4, 1))
    divider = TimeSeriesRankDivider(
        subdomain_layout=(2, 2), rank_dims=["x", "y"], rank_extent=[4, 4], overlap=1,
    )
    bottom_right_with_overlap = divider.get_subdomain_tensor_slice(
        arr, 3, with_overlap=True,
    )
    np.testing.assert_array_equal(
        bottom_right_with_overlap[0, :, :, 0],
        np.array([[1, 2, 0], [3, 4, 0], [0, 0, 0]]),
    )

    bottom_right_no_overlap = divider.get_subdomain_tensor_slice(
        arr, 3, with_overlap=False,
    )
    np.testing.assert_array_equal(bottom_right_no_overlap[0, :, :, 0], np.array([[4]]))


def test_RankDivider_subdomain_tensor_slice_overlap():
    xy_arr = np.pad(np.arange(1, 5).reshape(2, 2), pad_width=1)
    arr = np.reshape(xy_arr, (4, 4, 1))
    divider = RankDivider(
        subdomain_layout=(2, 2), rank_dims=["x", "y"], rank_extent=[4, 4], overlap=1,
    )
    bottom_right_with_overlap = divider.get_subdomain_tensor_slice(
        arr, 3, with_overlap=True,
    )
    np.testing.assert_array_equal(
        bottom_right_with_overlap[:, :, 0], np.array([[1, 2, 0], [3, 4, 0], [0, 0, 0]]),
    )

    bottom_right_no_overlap = divider.get_subdomain_tensor_slice(
        arr, 3, with_overlap=False,
    )
    np.testing.assert_array_equal(bottom_right_no_overlap[:, :, 0], np.array([[4]]))


def test_TimeSeriesRankDivider_get_subdomain_tensor_slice_covers_all_subdomains():
    divider = TimeSeriesRankDivider(
        subdomain_layout=(2, 2), rank_dims=["x", "y"], rank_extent=[2, 2], overlap=0,
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


def test_RankDivider_get_subdomain_tensor_slice_covers_all_subdomains():
    divider = RankDivider(
        subdomain_layout=(2, 2), rank_dims=["x", "y"], rank_extent=[2, 2], overlap=0,
    )
    arr = np.arange(1, 5).reshape(2, 2, 1)
    subdomain_values = []
    for s in range(divider.n_subdomains):
        subdomain_values.append(
            divider.get_subdomain_tensor_slice(arr, s, with_overlap=False)
            .flatten()
            .item()
        )
    assert set(subdomain_values) == {1, 2, 3, 4}


@pytest.mark.parametrize(
    "data_extent, overlap, with_overlap, ",
    [
        ([5, 6, 6], 1, True),
        ([5, 6, 6], 1, True),
        ([5, 6, 6], 1, False),
        ([5, 6, 6], 0, True),
        ([1, 4, 4], 0, False),
    ],
)
def test_TimeSeriesRankDivider_unstack_subdomain(data_extent, overlap, with_overlap):
    divider = TimeSeriesRankDivider(
        subdomain_layout=(2, 2),
        rank_dims=["x", "y"],
        rank_extent=data_extent[1:],
        overlap=overlap,
    )
    data_arr = np.arange(np.prod(data_extent)).reshape(*data_extent)
    subdomain_arr = divider.get_subdomain_tensor_slice(
        data_arr, 0, with_overlap=with_overlap,
    )
    stacked = stack_data(subdomain_arr, keep_first_dim=True)
    assert len(stacked.shape) == 2
    np.testing.assert_array_equal(
        divider.unstack_subdomain(stacked, with_overlap=with_overlap), subdomain_arr,
    )


@pytest.mark.parametrize(
    "data_extent, overlap, with_overlap, nz ",
    [
        ([6, 6], 1, True, 2),
        ([6, 6], 1, True, 2),
        ([4, 4], 0, False, 2),
        ([6, 6], 1, True, 1),
    ],
)
def test_RankDivider_unstack_subdomain(data_extent, overlap, with_overlap, nz):
    divider = RankDivider(
        subdomain_layout=(2, 2),
        rank_dims=["x", "y"],
        rank_extent=data_extent,
        overlap=overlap,
    )
    xy_shape = data_extent[:2]
    data_shape = (*xy_shape, nz) if nz > 1 else xy_shape
    data_arr = np.random.rand(*data_shape)
    subdomain_arr = divider.get_subdomain_tensor_slice(
        data_arr, 0, with_overlap=with_overlap,
    )
    stacked = stack_data(subdomain_arr, keep_first_dim=False)
    assert len(stacked.shape) == 1
    np.testing.assert_array_equal(
        divider.unstack_subdomain(stacked, with_overlap=with_overlap), subdomain_arr,
    )


def test_TimeSeriesRankDivider_flatten_subdomains_to_columns():
    nx, ny = 6, 6
    divider = TimeSeriesRankDivider(
        subdomain_layout=(2, 2), rank_dims=["x", "y"], rank_extent=[nx, ny], overlap=1,
    )
    input = np.array(
        [[0, 1, 10, 11], [2, 3, 12, 13], [20, 21, 30, 31], [22, 23, 32, 33]]
    )
    input_with_halo = np.pad(input, pad_width=1)
    # add time dim of length 2
    input_with_halo = np.stack([input_with_halo, input_with_halo], axis=0)
    flattened = divider.flatten_subdomains_to_columns(
        input_with_halo, with_overlap=True,
    )

    # 4 subdomains each with x, y dims (4, 4)
    # subdomain layout ranks are [[0, 2],[1, 3]] on square
    assert flattened.shape == (2, 16, 4)
    # subdomain 0
    np.testing.assert_array_almost_equal(
        flattened[0, :, 0],
        np.array([0, 0, 0, 0, 0, 0, 1, 10, 0, 2, 3, 12, 0, 20, 21, 30]),
    )

    # subdomain 2
    np.testing.assert_array_almost_equal(
        flattened[0, :, 2],
        np.array([0, 0, 0, 0, 1, 10, 11, 0, 3, 12, 13, 0, 21, 30, 31, 0]),
    )
    # subdomain 1
    np.testing.assert_array_almost_equal(
        flattened[0, :, 1],
        np.array([0, 2, 3, 12, 0, 20, 21, 30, 0, 22, 23, 32, 0, 0, 0, 0]),
    )

    # subdomain 2
    np.testing.assert_array_almost_equal(
        flattened[0, :, 3],
        np.array([3, 12, 13, 0, 21, 30, 31, 0, 23, 32, 33, 0, 0, 0, 0, 0]),
    )


def test_RankDivider_flatten_subdomains_to_columns():
    nx, ny = 6, 6
    divider = RankDivider(
        subdomain_layout=(2, 2), rank_dims=["x", "y"], rank_extent=[nx, ny], overlap=1,
    )
    input = np.array(
        [[0, 1, 10, 11], [2, 3, 12, 13], [20, 21, 30, 31], [22, 23, 32, 33]]
    )
    input_with_halo = np.pad(input, pad_width=1)
    flattened = divider.flatten_subdomains_to_columns(
        input_with_halo, with_overlap=True,
    )

    # 4 subdomains each with x, y dims (4, 4)
    # subdomain layout ranks are [[0, 2],[1, 3]] on square
    assert flattened.shape == (16, 4)
    # subdomain 0
    np.testing.assert_array_almost_equal(
        flattened[:, 0],
        np.array([0, 0, 0, 0, 0, 0, 1, 10, 0, 2, 3, 12, 0, 20, 21, 30]),
    )

    # subdomain 2
    np.testing.assert_array_almost_equal(
        flattened[:, 2],
        np.array([0, 0, 0, 0, 1, 10, 11, 0, 3, 12, 13, 0, 21, 30, 31, 0]),
    )
    # subdomain 1
    np.testing.assert_array_almost_equal(
        flattened[:, 1],
        np.array([0, 2, 3, 12, 0, 20, 21, 30, 0, 22, 23, 32, 0, 0, 0, 0]),
    )

    # subdomain 2
    np.testing.assert_array_almost_equal(
        flattened[:, 3],
        np.array([3, 12, 13, 0, 21, 30, 31, 0, 23, 32, 33, 0, 0, 0, 0, 0]),
    )


@pytest.mark.parametrize(
    "rank_extent,  overlap, expected_extent",
    [([8, 8], 1, [6, 6]), ([8, 8], 3, [2, 2]), ([8, 8], 0, [8, 8])],
)
def test_RankDivider__rank_extent_without_overlap(
    rank_extent, overlap, expected_extent
):
    divider = RankDivider(
        subdomain_layout=[2, 2],
        rank_dims=["x", "y"],
        rank_extent=rank_extent,
        overlap=overlap,
    )

    assert divider.rank_extent_without_overlap == expected_extent


def test_RankDivider_subdomain_xy_size_without_overlap():
    nx, ny = 8, 8
    divider = RankDivider(
        subdomain_layout=(2, 2), rank_dims=["x", "y"], rank_extent=[nx, ny], overlap=2,
    )
    assert divider.subdomain_xy_size_without_overlap == 2


def test_RankDivider_merge_subdomains():
    # Original (x, y, z) dims are (4, 4, 2)
    horizontal_array = np.arange(16).reshape(4, 4)
    data_orig = np.stack([horizontal_array, -1.0 * horizontal_array], axis=-1)
    rank_divider = RankDivider(
        subdomain_layout=(2, 2), rank_dims=["x", "y"], rank_extent=(4, 4), overlap=0,
    )

    # 'prediction' will just be the subdomains reshaped into columns and
    # concatenated together. We want the `merge_subdomains` function to
    # be able to take this 1D array and reshape it into the correct (x,y,z)
    # dimensions matching the original data.
    subdomain_columns = rank_divider.flatten_subdomains_to_columns(
        data_orig, with_overlap=False
    )
    prediction = np.concatenate(
        [subdomain_columns[:, s] for s in range(rank_divider.n_subdomains)], axis=-1
    )

    merged = rank_divider.merge_subdomains(prediction)
    np.testing.assert_array_equal(merged, data_orig)


@pytest.mark.parametrize(
    "original_dims, reordered_dims",
    [
        (["time", "x", "y", "z"], ["time", "x", "y", "z"]),
        (["time", "y", "x", "z"], ["time", "x", "y", "z"]),
    ],
)
def test__transpose_xy_dims(original_dims, reordered_dims):
    da = xr.DataArray(np.random.rand(5, 7, 7, 8), dims=original_dims)
    assert list(_transpose_xy_dims(da, rank_dims=["x", "y"]).dims) == reordered_dims
