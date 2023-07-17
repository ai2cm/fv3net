import numpy as np
import pytest
from fv3fit.reservoir.domain2 import (
    _check_feature_dims_consistent,
    RankXYDivider,
    OverlapRankXYDivider,
)


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

    return np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]])


def test_get_subdomain():

    rank_domain = get_4x4_rank_domain()

    # Test with valid input
    divider = RankXYDivider((2, 2), (4, 4))
    subdomain = divider.get_subdomain(rank_domain, 0)
    assert np.all(subdomain == np.array([[0, 1], [4, 5]]))

    subdomain = divider.get_subdomain(rank_domain, 3)
    assert np.all(subdomain == np.array([[10, 11], [14, 15]]))

    with pytest.raises(ValueError):
        divider.get_subdomain(rank_domain, 4)

    with pytest.raises(ValueError):
        divider.get_subdomain(rank_domain[0:2], 0)


def test_get_subdomain_with_feature():

    rank_domain = get_4x4_rank_domain()
    stacked = np.concatenate([rank_domain[..., None] + i for i in range(3)], axis=-1)

    # Test with valid input
    divider = RankXYDivider((2, 2), (4, 4), z_feature=3)
    subdomain = divider.get_subdomain(stacked, 0)
    assert subdomain.shape == (2, 2, 3)
    assert np.all(subdomain[..., 0] == np.array([[0, 1], [4, 5]]))
    assert np.all(subdomain[..., 2] == np.array([[2, 3], [6, 7]]))

    subdomain = divider.get_subdomain(stacked, 3)
    assert np.all(subdomain[..., 0] == np.array([[10, 11], [14, 15]]))


def test_get_subdomain_with_leading():

    rank_domain = get_4x4_rank_domain()
    stacked = np.concatenate([rank_domain[None] + i for i in range(3)], axis=0)

    # Test with valid input
    divider = RankXYDivider((2, 2), (4, 4))
    subdomain = divider.get_subdomain(stacked, 0)
    assert subdomain.shape == (3, 2, 2)
    assert np.all(subdomain[0] == np.array([[0, 1], [4, 5]]))
    assert np.all(subdomain[2] == np.array([[2, 3], [6, 7]]))

    subdomain = divider.get_subdomain(stacked, 3)
    assert np.all(subdomain[0] == np.array([[10, 11], [14, 15]]))


def test_get_all_subdomains():
    rank_domain = get_4x4_rank_domain()

    divider = RankXYDivider((2, 2), (4, 4))
    all_subdomains = divider.get_all_subdomains(rank_domain)

    assert len(all_subdomains.shape) == 3
    assert len(all_subdomains) == 4
    assert np.all(all_subdomains[0] == np.array([[0, 1], [4, 5]]))


def test_flatten_subdomain_features():
    # Test with valid input
    divider = RankXYDivider((2, 2), (10, 20), 3)
    data = np.ones((5, 10, 3))
    flattened = divider.flatten_subdomain_features(data)
    assert flattened.shape == (150,)

    # Test with invalid input
    data = np.random.rand(5, 10, 4)
    with pytest.raises(ValueError):
        divider.flatten_subdomain_features(data)


def test_reshape_flat_subdomain_features():
    # Test with valid input
    divider = RankXYDivider((2, 2), (10, 20), 3)
    data = np.ones((150))
    reshaped = divider.reshape_flat_subdomain_features(data)
    assert reshaped.shape == (5, 10, 3)

    # Test with invalid input
    data = np.ones((50, 4))
    with pytest.raises(ValueError):
        divider.reshape_flat_subdomain_features(data)


def test_merge_all_subdomains():
    # Test with valid input
    divider = RankXYDivider((2, 2), (10, 20), 3)
    data = np.ones((4, 5, 10, 3))
    merged = divider.merge_all_subdomains(data)
    assert merged.shape == (10, 20, 3)

    # Test with invalid input
    data = np.ones((3, 5, 10, 3))
    with pytest.raises(ValueError):
        divider.merge_all_subdomains(data)


def test_all_subdomain_merge_roundtrip():
    # Test with valid input
    divider = RankXYDivider((2, 2), (10, 20), 3)
    data = np.random.rand(10, 20, 3)
    divided = divider.get_all_subdomains(data)
    divided_flat = divider.flatten_subdomain_features(divided)
    divided_reshaped = divider.reshape_flat_subdomain_features(divided_flat)
    merged = divider.merge_all_subdomains(divided_reshaped)
    assert np.all(merged == data)


def test_get_overlap_subdomain():

    rank_domain = get_4x4_rank_domain()

    # Test with valid input
    divider = OverlapRankXYDivider((2, 2), (4, 4), overlap=1)
    subdomain = divider.get_subdomain(rank_domain, 0)
    assert np.all(subdomain == np.array([[0, 1, 2], [4, 5, 6], [8, 9, 10]]))

    subdomain = divider.get_subdomain(rank_domain, 3)
    assert np.all(subdomain == np.array([[5, 6, 7], [9, 10, 11], [13, 14, 15]]))

    with pytest.raises(ValueError):
        divider.get_subdomain(rank_domain, 4)

    with pytest.raises(ValueError):
        divider.get_subdomain(rank_domain[0:2], 0)
