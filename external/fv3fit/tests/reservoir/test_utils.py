import numpy as np
import pytest
from fv3fit.reservoir.utils import (
    square_even_terms,
    process_batch_data,
    SynchronziationTracker,
    assure_txyz_dims,
)

from fv3fit.reservoir.transformers import DoNothingAutoencoder
from fv3fit.reservoir.domain2 import RankXYDivider


def test_SynchronziationTracker():
    sync_tracker = SynchronziationTracker(n_synchronize=6)
    batches = np.arange(15).reshape(3, 5)
    expected = [np.array([]), np.array([6, 7, 8, 9]), np.array([10, 11, 12, 13, 14])]
    for expected_trimmed, batch in zip(expected, batches):
        sync_tracker.count_synchronization_steps(len(batch))
        np.testing.assert_array_equal(
            sync_tracker.trim_synchronization_samples_if_needed(batch), expected_trimmed
        )


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


@pytest.mark.parametrize(
    "nz, overlap, trim_halo",
    [(1, 1, False), (3, 1, False), (1, 0, False), (3, 0, False), (3, 1, True)],
)
def test_process_batch_data(nz, overlap, trim_halo):
    nt, nx, ny = 10, 8, 8
    subdomain_layout = (2, 2)
    autoencoder = DoNothingAutoencoder([nz, nz])
    batch_data = {
        "a": np.ones((nt, nx, ny, nz)),
        "b": np.ones((nt, nx, ny, nz)),
    }
    rank_divider = RankXYDivider(
        subdomain_layout=subdomain_layout,
        overlap=overlap,
        overlap_rank_extent=(nx, ny),
        z_feature_size=autoencoder.n_latent_dims,
    )

    time_series = process_batch_data(
        variables=["a", "b"],
        batch_data=batch_data,
        rank_divider=rank_divider,
        autoencoder=autoencoder,
        trim_halo=trim_halo,
    )

    if trim_halo is True:
        features_per_subdomain = (
            rank_divider.get_no_overlap_rank_divider().flat_subdomain_len
        )
    else:
        features_per_subdomain = rank_divider.flat_subdomain_len
    assert time_series.shape == (nt, rank_divider.n_subdomains, features_per_subdomain,)


@pytest.mark.parametrize(
    "encoder_shape",
    [(8, 8, 3), (6, 6, 3)],
    ids=["encoder_expects_halo_input", "encoder_expects_no_halo_input"],
)
def test_process_batch_data_trim_workaround(encoder_shape):
    """
    This is a temporary test for hanlding the processing when
    an autoencoder may or may not expect trimmed input data.
    Once we resolve this issue, this test should be removed.
    Need some sort of standard on what type of data encoders
    expect as inputs.
    """

    class TrimCheckAutoencoder:
        def __init__(self, expected_dimensions):
            self.expected_dimensions = expected_dimensions

        def encode_txyz(self, data):
            if data[0].shape[-3:] != self.expected_dimensions:
                raise ValueError(
                    f"Expected data shape {self.expected_dimensions} "
                    f"but got {data[0].shape[-3:]}"
                )
            return data[0]

    nt, nx, ny, nz = 10, 8, 8, 3
    overlap = 1
    subdomain_layout = (2, 2)
    batch_data = {
        "a": np.ones((nt, nx, ny, nz)),
    }
    rank_divider = RankXYDivider(
        subdomain_layout=subdomain_layout,
        overlap=overlap,
        overlap_rank_extent=(nx, ny),
        z_feature_size=nz,
    )

    encoder = TrimCheckAutoencoder(encoder_shape)

    time_series = process_batch_data(
        variables=["a"],
        batch_data=batch_data,
        rank_divider=rank_divider,
        autoencoder=encoder,
        trim_halo=True,
    )
    expected_shape = (
        nt,
        rank_divider.n_subdomains,
        rank_divider.get_no_overlap_rank_divider().flat_subdomain_len,
    )
    assert time_series.shape == expected_shape


def test_assure_txyz_dims():
    nt, nx, ny, nz = 5, 4, 4, 6
    arr_3d = np.ones((nt, nx, ny, nz))
    arr_2d = np.ones((nt, nx, ny))
    assert assure_txyz_dims(arr_3d).shape == (nt, nx, ny, nz)
    assert assure_txyz_dims(arr_2d).shape == (nt, nx, ny, 1)


def test_assure_txyz_dims_incompatible_shapes():
    nt, nx, ny, nz = 5, 4, 4, 6
    with pytest.raises(ValueError):
        assure_txyz_dims(np.ones((nt, nx, ny, nz, 2)))
