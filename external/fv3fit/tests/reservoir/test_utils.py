import numpy as np
import pytest
from fv3fit.reservoir.utils import square_even_terms, process_batch_Xy_data
from fv3fit.reservoir.transformers import DoNothingAutoencoder
from fv3fit.reservoir.domain import RankDivider


@pytest.mark.parametrize(
    "arr, axis, expected",
    [
        (np.arange(4), 0, np.array([0, 1, 4, 3])),
        (np.arange(4).reshape(1, -1), 1, np.array([[0, 1, 4, 3]])),
        (np.arange(8).reshape(2, 4), 0, np.array([[0, 1, 4, 9], [4, 5, 6, 7]])),
        (
            np.arange(10).reshape(2, 5),
            1,
            np.array([[0, 1, 4, 3, 16], [25, 6, 49, 8, 81]]),
        ),
    ],
)
def test__square_even_terms(arr, axis, expected):
    np.testing.assert_array_equal(square_even_terms(arr, axis=axis), expected)


@pytest.mark.parametrize("nz", [1, 3])
def test_process_batch_Xy_data(nz):
    overlap = 1
    nvars = 2
    nt, nx, ny = 10, 8, 8
    subdomain_layout = [2, 2]
    rank_divider = RankDivider(
        subdomain_layout=subdomain_layout,
        rank_dims=["x", "y"],
        rank_extent=[nx, ny],
        overlap=overlap,
    )
    autoencoder = DoNothingAutoencoder([nz, nz])
    batch_data = {
        "a": np.ones((nt, nx, ny, nz)),
        "b": np.ones((nt, nx, ny, nz)),
    }
    time_series_with_overlap, time_series_without_overlap = process_batch_Xy_data(
        variables=["a", "b"],
        batch_data=batch_data,
        input_rank_divider=rank_divider,
        autoencoder=autoencoder,
    )
    features_per_subdomain_with_overlap = (
        np.prod(rank_divider.get_subdomain_extent(True)) * nvars * nz
    )
    features_per_subdomain_without_overlap = (
        np.prod(rank_divider.get_subdomain_extent(False)) * nvars * nz
    )
    assert time_series_with_overlap.shape == (
        nt,
        features_per_subdomain_with_overlap,
        rank_divider.n_subdomains,
    )
    assert time_series_without_overlap.shape == (
        nt,
        features_per_subdomain_without_overlap,
        rank_divider.n_subdomains,
    )
