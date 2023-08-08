import numpy as np
import pytest
from fv3fit.reservoir.utils import square_even_terms, process_batch_Xy_data
from fv3fit.reservoir.transformers import DoNothingAutoencoder
from fv3fit.reservoir.domain2 import RankXYDivider


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
    nt, nx, ny = 10, 8, 8
    subdomain_layout = (2, 2)
    autoencoder = DoNothingAutoencoder([nz, nz])
    batch_data = {
        "a": np.ones((nt, nx, ny, nz)),
        "b": np.ones((nt, nx, ny, nz)),
    }
    overlap_rank_divider = RankXYDivider(
        subdomain_layout=subdomain_layout,
        overlap=overlap,
        overlap_rank_extent=(nx, ny),
        z_feature=autoencoder.n_latent_dims,
    )
    rank_divider = overlap_rank_divider.get_no_overlap_rank_divider()

    time_series_with_overlap, time_series_without_overlap = process_batch_Xy_data(
        variables=["a", "b"],
        batch_data=batch_data,
        rank_divider=overlap_rank_divider,
        autoencoder=autoencoder,
    )
    features_per_subdomain_with_overlap = overlap_rank_divider.flat_subdomain_len
    features_per_subdomain_without_overlap = rank_divider.flat_subdomain_len
    assert time_series_with_overlap.shape == (
        nt,
        overlap_rank_divider.n_subdomains,
        features_per_subdomain_with_overlap,
    )
    assert time_series_without_overlap.shape == (
        nt,
        rank_divider.n_subdomains,
        features_per_subdomain_without_overlap,
    )
