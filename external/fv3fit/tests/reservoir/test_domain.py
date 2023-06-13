import numpy as np
import pytest
from fv3fit.reservoir.domain import (
    slice_along_axis,
    RankDivider,
    assure_same_dims,
)
from fv3fit.reservoir._reshaping import stack_samples


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
    "rank_dims": ["x", "y", "z"],
    "rank_extent": [6, 6, 7],
}


def test_assure_same_dims():
    nt, nx, ny, nz = 5, 4, 4, 6
    arr_3d = np.ones((nt, nx, ny, nz))
    arr_2d = np.ones((nt, nx, ny,))
    data = [arr_3d, arr_2d]
    assert assure_same_dims(data)[0].shape == (nt, nx, ny, nz)
    assert assure_same_dims(data)[1].shape == (nt, nx, ny, 1)


def test_assure_same_dims_incompatible_shapes():
    nt, nx, ny, nz = 5, 4, 4, 6
    arr_3d = np.ones((nt, nx, ny, nz, 2))
    arr_2d = np.ones((nt, nx, ny,))
    data = [arr_3d, arr_2d]
    with pytest.raises(ValueError):
        assure_same_dims(data)


@pytest.mark.parametrize(
    "layout, rank_extent, overlap, expected_with_overlap, expected_without_overlap",
    [
        [(2, 2), (6, 6, 2), 0, (3, 3, 2), (3, 3, 2)],
        [(2, 2), (6, 6, 2), 1, (4, 4, 2), (2, 2, 2)],
        [(2, 2), (10, 10, 2), 2, (7, 7, 2), (3, 3, 2)],
    ],
)
def test_RankDivider_get_subdomain_extent(
    layout, rank_extent, overlap, expected_with_overlap, expected_without_overlap
):
    divider = RankDivider(
        subdomain_layout=layout,
        rank_dims=["x", "y", "z"],
        rank_extent=rank_extent,
        overlap=overlap,
    )
    assert divider.get_subdomain_extent(with_overlap=True) == expected_with_overlap
    assert divider.get_subdomain_extent(with_overlap=False) == expected_without_overlap


def test_RankDivider_subdomain_slice():
    nz = 2
    divider = RankDivider(
        subdomain_layout=(2, 2),
        rank_dims=["x", "y", "z"],
        rank_extent=[6, 6, nz],
        overlap=1,
    )
    z_slice = slice(0, nz)

    assert divider.subdomain_slice(0, with_overlap=True) == (
        slice(0, 4),
        slice(0, 4),
        z_slice,
    )
    assert divider.subdomain_slice(0, with_overlap=False) == (
        slice(1, 3),
        slice(1, 3),
        z_slice,
    )
    assert divider.subdomain_slice(3, with_overlap=True) == (
        slice(2, 6),
        slice(2, 6),
        z_slice,
    )
    assert divider.subdomain_slice(3, with_overlap=False) == (
        slice(3, 5),
        slice(3, 5),
        z_slice,
    )


def test_RankDivider_subdomain_tensor_slice_overlap_with_time_dim():
    xy_arr = np.pad(np.arange(1, 5).reshape(2, 2), pad_width=1)
    arr = np.reshape(xy_arr, (1, 4, 4, 1))
    divider = RankDivider(
        subdomain_layout=(2, 2),
        rank_dims=["x", "y", "z"],
        rank_extent=[4, 4, 1],
        overlap=1,
    )
    bottom_right_with_overlap = divider.get_subdomain_tensor_slice(
        arr, 3, with_overlap=True, data_has_time_dim=True
    )
    np.testing.assert_array_equal(
        bottom_right_with_overlap[0, :, :, 0],
        np.array([[1, 2, 0], [3, 4, 0], [0, 0, 0]]),
    )

    bottom_right_no_overlap = divider.get_subdomain_tensor_slice(
        arr, 3, with_overlap=False, data_has_time_dim=True
    )
    np.testing.assert_array_equal(bottom_right_no_overlap[0, :, :, 0], np.array([[4]]))


def test_RankDivider_subdomain_tensor_slice_overlap_no_time_dim():
    xy_arr = np.pad(np.arange(1, 5).reshape(2, 2), pad_width=1)
    arr = np.reshape(xy_arr, (4, 4, 1))
    divider = RankDivider(
        subdomain_layout=(2, 2),
        rank_dims=["x", "y", "z"],
        rank_extent=[4, 4, 1],
        overlap=1,
    )
    bottom_right_with_overlap = divider.get_subdomain_tensor_slice(
        arr, 3, with_overlap=True, data_has_time_dim=False
    )
    np.testing.assert_array_equal(
        bottom_right_with_overlap[:, :, 0], np.array([[1, 2, 0], [3, 4, 0], [0, 0, 0]]),
    )

    bottom_right_no_overlap = divider.get_subdomain_tensor_slice(
        arr, 3, with_overlap=False, data_has_time_dim=False
    )
    np.testing.assert_array_equal(bottom_right_no_overlap[:, :, 0], np.array([[4]]))


def test_RankDivider_get_subdomain_tensor_slice():
    divider = RankDivider(
        subdomain_layout=(2, 2),
        rank_dims=["x", "y", "z"],
        rank_extent=[2, 2, 1],
        overlap=0,
    )
    arr = np.arange(1, 5).reshape(1, 2, 2, 1)
    subdomain_values = []
    for s in range(divider.n_subdomains):
        subdomain_values.append(
            divider.get_subdomain_tensor_slice(
                arr, s, with_overlap=False, data_has_time_dim=True
            )
            .flatten()
            .item()
        )

    assert set(subdomain_values) == {1, 2, 3, 4}


@pytest.mark.parametrize(
    "data_extent, overlap, with_overlap, ",
    [
        ([5, 6, 6, 1], 1, True),
        ([5, 6, 6, 2], 1, True),
        ([5, 6, 6, 2], 1, False),
        ([5, 6, 6, 2], 0, True),
        ([1, 4, 4, 3], 0, False),
        ([6, 6, 1], 1, True),
        ([4, 4, 3], 0, False),
    ],
)
def test_RankDivider_unstack_subdomain(data_extent, overlap, with_overlap):
    if len(data_extent) == 4:
        data_has_time_dim = True
        rank_extent = data_extent[1:]
    else:
        data_has_time_dim = False
        rank_extent = data_extent

    divider = RankDivider(
        subdomain_layout=(2, 2),
        rank_dims=["x", "y", "z"],
        rank_extent=rank_extent,
        overlap=overlap,
    )
    data_arr = np.arange(np.prod(data_extent)).reshape(*data_extent)
    subdomain_arr = divider.get_subdomain_tensor_slice(
        data_arr, 0, with_overlap=with_overlap, data_has_time_dim=data_has_time_dim
    )
    stacked = stack_samples(subdomain_arr, data_has_time_dim=data_has_time_dim)
    if data_has_time_dim is True:
        assert len(stacked.shape) == 2
    else:
        assert len(stacked.shape) == 1
    np.testing.assert_array_equal(
        divider.unstack_subdomain(
            stacked, with_overlap=with_overlap, data_has_time_dim=data_has_time_dim
        ),
        subdomain_arr,
    )


def test_RankDivider_flatten_subdomains_to_columns():
    nx, ny, nz = 6, 6, 2
    divider = RankDivider(
        subdomain_layout=(2, 2),
        rank_dims=["x", "y", "z"],
        rank_extent=[nx, ny, nz],
        overlap=1,
    )
    input = np.array(
        [[0, 1, 10, 11], [2, 3, 12, 13], [20, 21, 30, 31], [22, 23, 32, 33]]
    )
    input_with_halo = np.pad(input, pad_width=1)
    # add time dim of length 2
    input_with_halo = np.stack([input_with_halo, input_with_halo], axis=0)
    flattened = divider.flatten_subdomains_to_columns(
        input_with_halo, with_overlap=True, data_has_time_dim=True
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


@pytest.mark.parametrize(
    "rank_extent,  overlap, expected_extent",
    [
        ([8, 8, 5], 1, [6, 6, 5]),
        ([8, 8, 2], 3, [2, 2, 2]),
        ([8, 8, 5], 0, [8, 8, 5]),
        ([8, 8, 2], 0, [8, 8, 2]),
    ],
)
def test_RankDivider__rank_extent_without_overlap(
    rank_extent, overlap, expected_extent
):
    divider = RankDivider(
        subdomain_layout=[2, 2],
        rank_dims=["x", "y", "z"],
        rank_extent=rank_extent,
        overlap=overlap,
    )

    assert divider._rank_extent_without_overlap == expected_extent


def test_RankDivider_subdomain_xy_size_without_overlap():
    nx, ny, nz = 8, 8, 2
    divider = RankDivider(
        subdomain_layout=(2, 2),
        rank_dims=["x", "y", "z"],
        rank_extent=[nx, ny, nz],
        overlap=2,
    )
    assert divider.subdomain_xy_size_without_overlap == 2
