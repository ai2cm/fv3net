import numpy as np
import pytest
import tempfile

from fv3fit.reservoir.domain2 import (
    _check_feature_dims_consistent,
    RankXYDivider,
)
from fv3fit.reservoir.domain import RankDivider


def test_check_feature_dims_consistent():
    # Test with consistent feature dimensions
    data_shape = (10, 20, 30, 4)
    feature_shape = (30, 4)
    _check_feature_dims_consistent(data_shape, feature_shape)

    # Test with inconsistent feature dimensions
    data_shape = (10, 20, 30, 4)
    feature_shape = (30, 5)
    with pytest.raises(ValueError):
        _check_feature_dims_consistent(data_shape, feature_shape)

    # Test with scalar feature dimensions
    data_shape = (10, 20, 30, 4)
    feature_shape = (4,)
    _check_feature_dims_consistent(data_shape, feature_shape)

    # Test with empty feature dimensions
    data_shape = (10, 20, 30, 4)
    feature_shape = ()
    with pytest.raises(ValueError):
        _check_feature_dims_consistent(data_shape, feature_shape)

    # Test with data shape smaller than feature shape
    data_shape = (10, 20, 4)
    feature_shape = (30, 4)
    with pytest.raises(ValueError):
        _check_feature_dims_consistent(data_shape, feature_shape)


def get_4x4_rank_domain():
    return np.arange(16).reshape((4, 4))


def test_rank_divider_init():
    # no overlap
    divider = RankXYDivider((2, 2), 0, rank_extent=(4, 4))
    assert divider.n_subdomains == 4
    assert divider.rank_extent == (4, 4)
    assert divider.overlap_rank_extent == (4, 4)

    # overlap
    divider = RankXYDivider((2, 2), 1, overlap_rank_extent=(6, 6))
    assert divider.n_subdomains == 4
    assert divider.rank_extent == (4, 4)
    assert divider.overlap_rank_extent == (6, 6)

    same_divider = RankXYDivider((2, 2), 1, rank_extent=(4, 4))
    assert divider == same_divider


def test_rank_divider_init_failures():
    # both rank and overlap extent specified
    with pytest.raises(ValueError):
        RankXYDivider((2, 2), 1, (4, 4), (6, 6))

    # neither rank nor overlap extent specified
    with pytest.raises(ValueError):
        RankXYDivider((2, 2), 1)

    # negative overlap
    with pytest.raises(ValueError):
        RankXYDivider((2, 2), -1, rank_extent=(4, 4))

    # 2D layout error
    with pytest.raises(ValueError):
        RankXYDivider((2, 2, 3), 0, rank_extent=(4, 4))

    # 2D extent error
    with pytest.raises(ValueError):
        RankXYDivider((2, 2), 0, rank_extent=(4, 4, 3))

    # extent divisibility
    with pytest.raises(ValueError):
        RankXYDivider((2, 2), 0, rank_extent=(3, 4))

    with pytest.raises(ValueError):
        RankXYDivider((2, 2), 0, rank_extent=(4, 3))

    with pytest.raises(ValueError):
        RankXYDivider((2, 2), 1, overlap_rank_extent=(4, 3))


def test_get_subdomain():

    rank_domain = get_4x4_rank_domain()

    # test basic subdomain upper left corner
    divider = RankXYDivider((2, 2), 0, rank_extent=(4, 4))
    subdomain = divider.get_subdomain(rank_domain, 0)
    np.testing.assert_equal(subdomain, np.array([[0, 1], [4, 5]]))

    # lower right corner
    subdomain = divider.get_subdomain(rank_domain, 3)
    np.testing.assert_equal(subdomain, np.array([[10, 11], [14, 15]]))

    with pytest.raises(ValueError):
        divider.get_subdomain(rank_domain, 4)

    with pytest.raises(ValueError):
        divider.get_subdomain(rank_domain[0:2], 0)


def test_get_subdomain_with_feature():

    rank_domain = get_4x4_rank_domain()
    stacked = np.concatenate([rank_domain[..., None] + i for i in range(3)], axis=-1)

    divider = RankXYDivider((2, 2), 0, rank_extent=(4, 4), z_feature_size=3)
    subdomain = divider.get_subdomain(stacked, 0)
    assert subdomain.shape == (2, 2, 3)
    np.testing.assert_equal(subdomain[..., 0], np.array([[0, 1], [4, 5]]))
    np.testing.assert_equal(subdomain[..., 2], np.array([[2, 3], [6, 7]]))

    subdomain = divider.get_subdomain(stacked, 3)
    np.testing.assert_equal(subdomain[..., 0], np.array([[10, 11], [14, 15]]))


def test_get_subdomain_with_leading():

    rank_domain = get_4x4_rank_domain()
    stacked = np.concatenate([rank_domain[None] + i for i in range(3)], axis=0)

    # check leading dimension is preserved while subdomains are consistent
    divider = RankXYDivider((2, 2), 0, rank_extent=(4, 4))
    subdomain = divider.get_subdomain(stacked, 0)
    assert subdomain.shape == (3, 2, 2)
    np.testing.assert_equal(subdomain[0], np.array([[0, 1], [4, 5]]))
    np.testing.assert_equal(subdomain[2], np.array([[2, 3], [6, 7]]))

    subdomain = divider.get_subdomain(stacked, 3)
    np.testing.assert_equal(subdomain[0], np.array([[10, 11], [14, 15]]))


def test_get_all_subdomains():
    rank_domain = get_4x4_rank_domain()

    divider = RankXYDivider((2, 2), 0, rank_extent=(4, 4))
    all_subdomains = divider.get_all_subdomains(rank_domain)

    assert len(all_subdomains.shape) == 3
    assert len(all_subdomains) == 4
    np.testing.assert_equal(all_subdomains[0], np.array([[0, 1], [4, 5]]))


def test_get_all_subdomains_with_leading():
    # Checks that subdomain axis is added in the proper location
    ntimes = 3
    rank_domain = np.array([get_4x4_rank_domain()] * ntimes)

    divider = RankXYDivider((2, 2), 0, rank_extent=(4, 4))
    all_subdomains = divider.get_all_subdomains(rank_domain)

    assert len(all_subdomains.shape) == 4
    assert len(all_subdomains) == 3
    assert all_subdomains.shape[1] == 4
    np.testing.assert_equal(all_subdomains[:, 0], np.array([[[0, 1], [4, 5]]] * ntimes))


def test_flatten_subdomain_features():
    divider = RankXYDivider((2, 2), 0, rank_extent=(10, 20), z_feature_size=3)
    data = np.ones((5, 10, 3))
    flattened = divider.flatten_subdomain_features(data)
    assert flattened.shape == (150,)

    # Incorrect data feature shape
    data = np.random.rand(5, 10, 4)
    with pytest.raises(ValueError):
        divider.flatten_subdomain_features(data)


def test_reshape_flat_subdomain_features():
    divider = RankXYDivider((2, 2), 0, rank_extent=(10, 20), z_feature_size=3)
    data = np.ones((150))
    reshaped = divider.reshape_flat_subdomain_features(data)
    assert reshaped.shape == (5, 10, 3)

    # Incorect data feature shape
    data = np.ones((50, 4))
    with pytest.raises(ValueError):
        divider.reshape_flat_subdomain_features(data)


def test_merge_all_subdomains():
    divider = RankXYDivider((2, 2), 0, rank_extent=(10, 20), z_feature_size=3)
    data = np.ones((4, 5, 10, 3))
    merged = divider.merge_all_subdomains(data)
    assert merged.shape == (10, 20, 3)

    # Incorrect data feature shape
    data = np.ones((3, 5, 10, 3))
    with pytest.raises(ValueError):
        divider.merge_all_subdomains(data)


def test_all_subdomain_merge_roundtrip():
    divider = RankXYDivider((2, 2), 0, rank_extent=(10, 20), z_feature_size=3)
    data = np.random.rand(15, 10, 20, 3)
    divided_flat = divider.get_all_subdomains_with_flat_feature(data)
    merged = divider.merge_all_flat_feature_subdomains(divided_flat)
    np.testing.assert_equal(merged, data)


def test_take_sequence_over_subdomains():
    divider = RankXYDivider((2, 2), 0, rank_extent=(4, 4))
    data = get_4x4_rank_domain()
    data_with_leading = np.array([data + 1 for i in range(3)])

    subs = divider.get_all_subdomains(data)
    subs_with_leading = divider.get_all_subdomains(data_with_leading)

    # no change
    res = divider.subdomains_to_leading_axis(subs)
    np.testing.assert_equal(res, subs)

    # leading time
    res = divider.subdomains_to_leading_axis(subs_with_leading)
    assert res.shape == (4, 3, 2, 2)
    np.testing.assert_equal(res[:, 1], subs_with_leading[1])

    # leading time and flat feature
    flat_subs_with_leading = divider.flatten_subdomain_features(subs_with_leading)
    res = divider.subdomains_to_leading_axis(flat_subs_with_leading, flat_feature=True)
    assert res.shape == (4, 3, 4)
    np.testing.assert_equal(res[:, 1], flat_subs_with_leading[1])


def test_get_overlap_subdomain():

    rank_domain = get_4x4_rank_domain()

    # Test 1x1 subdomains with overlap 1
    divider = RankXYDivider((2, 2), 1, overlap_rank_extent=(4, 4))
    subdomain = divider.get_subdomain(rank_domain, 0)
    np.testing.assert_equal(subdomain, np.array([[0, 1, 2], [4, 5, 6], [8, 9, 10]]))

    subdomain = divider.get_subdomain(rank_domain, 3)
    np.testing.assert_equal(subdomain, np.array([[5, 6, 7], [9, 10, 11], [13, 14, 15]]))

    with pytest.raises(ValueError):
        divider.get_subdomain(rank_domain, 4)

    with pytest.raises(ValueError):
        divider.get_subdomain(rank_domain[0:2], 0)


def test_merge_subdomain_with_overlap_fails():
    divider = RankXYDivider((2, 2), 1, overlap_rank_extent=(4, 4))
    data = get_4x4_rank_domain()
    subdomains = divider.get_all_subdomains(data)
    with pytest.raises(ValueError):
        divider.merge_all_subdomains(subdomains)


@pytest.mark.parametrize(
    "overlap, z_feature", [(0, None), (0, 3), (1, None), (1, 3),],
)
def test_dump_load(overlap, z_feature):
    divider = RankXYDivider(
        (2, 2), overlap, rank_extent=(4, 4), z_feature_size=z_feature
    )
    with tempfile.NamedTemporaryFile() as tmp:
        divider.dump(tmp.name)
        loaded = RankXYDivider.load(tmp.name)
        assert divider == loaded


@pytest.mark.parametrize(
    "ntimes, z_features",
    [(0, None), (1, None), (2, None), (0, 1), (0, 2), (1, 1), (2, 2)],
)
def test_trim_overlap_rank(ntimes, z_features):
    overlap = 1
    rank_data = get_4x4_rank_domain()
    trimmed = rank_data[overlap:-overlap, overlap:-overlap]
    if ntimes > 0:
        rank_data = np.array([rank_data + i for i in range(ntimes)])
        trimmed = np.array([trimmed + i for i in range(ntimes)])
    if z_features is not None:
        rank_data = np.stack([rank_data] * z_features, axis=-1)
        trimmed = np.stack([trimmed] * z_features, axis=-1)

    divider = RankXYDivider(
        (2, 2), overlap, overlap_rank_extent=(4, 4), z_feature_size=z_features
    )
    result = divider.trim_halo_from_rank_data(rank_data)

    np.testing.assert_equal(result, trimmed)


# TODO: used as a direct comparison, delete when no longer needed
def test_subdomain_decomp_against_original_RankDivider():
    original = RankDivider((2, 2), ["x", "y"], (4, 4), overlap=1)
    new = RankXYDivider((2, 2), 1, overlap_rank_extent=(4, 4))

    rank_domain = get_4x4_rank_domain()

    # returns flat_feature, subdomain
    orig_sub_flat = original.flatten_subdomains_to_columns(
        rank_domain, with_overlap=True
    )

    # returns subdomain, flat_feature
    new_sub_flat = new.get_all_subdomains_with_flat_feature(rank_domain)

    np.testing.assert_equal(orig_sub_flat.T, new_sub_flat)
