from fv3fit.pytorch.cyclegan.cyclegan_trainer import ResultsAggregator, get_percentile
from typing import Tuple
import pytest
import numpy as np


@pytest.mark.parametrize(
    "bins, hist, pct, expected",
    [
        pytest.param(
            np.array([0, 1]), np.array([10]), 0.8, 0.8, id="single_bin_interpolation"
        ),
        pytest.param(
            np.array([0, 1, 2]),
            np.array([10, 10]),
            0.8,
            1.6,
            id="two_bin_interpolation",
        ),
        pytest.param(
            np.array([0, 1, 2]), np.array([10, 10]), 1.0, 2.0, id="two_bin_end_value"
        ),
    ],
)
def test_get_percentile(bins, hist, pct, expected):
    result = get_percentile(bins, hist, pct)
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("shape", [(1, 1, 1, 1, 1), (2, 2, 2, 2, 2)])
@pytest.mark.parametrize("n_samples", [1, 5])
def test_results_aggregator_means(shape: Tuple[int], n_samples: int):
    results_real_a = np.random.randn(n_samples, *shape)
    results_real_b = np.random.randn(n_samples, *shape)
    results_fake_a = np.random.randn(n_samples, *shape)
    results_fake_b = np.random.randn(n_samples, *shape)
    aggregator = ResultsAggregator(histogram_vmax=100.0)
    for i in range(n_samples):
        aggregator.record_results(
            real_a=results_real_a[i],
            real_b=results_real_b[i],
            fake_a=results_fake_a[i],
            fake_b=results_fake_b[i],
        )
    np.testing.assert_almost_equal(
        aggregator.mean_real_a, np.mean(results_real_a, axis=(0, 1))
    )
    np.testing.assert_almost_equal(
        aggregator.mean_real_b, np.mean(results_real_b, axis=(0, 1))
    )
    np.testing.assert_almost_equal(
        aggregator.mean_fake_a, np.mean(results_fake_a, axis=(0, 1))
    )
    np.testing.assert_almost_equal(
        aggregator.mean_fake_b, np.mean(results_fake_b, axis=(0, 1))
    )


def test_results_aggregator_histogram_consistency():
    n_epochs = 10
    n_samples = 10
    n_tiles = 6
    n_channel = 3
    nx = 4
    ny = 5
    shape = (n_epochs, n_samples, n_tiles, n_channel, nx, ny)
    results_real_a = np.random.randn(*shape)
    results_real_b = np.random.randn(*shape)
    results_fake_a = np.random.randn(*shape)
    results_fake_b = np.random.randn(*shape)
    aggregator = ResultsAggregator(histogram_vmax=100.0)
    for i in range(n_epochs):
        aggregator.record_results(
            real_a=results_real_a[i],
            real_b=results_real_b[i],
            fake_a=results_fake_a[i],
            fake_b=results_fake_b[i],
        )
    for hist in (
        aggregator.fake_a_histogram,
        aggregator.fake_b_histogram,
        aggregator.real_a_histogram,
        aggregator.real_b_histogram,
    ):
        assert hist.shape == (n_channel, 100)  # 100 bins
        assert np.all(hist >= 0)
        assert np.all(hist <= 1)
        np.testing.assert_almost_equal(
            np.sum(hist * np.diff(aggregator.bins), axis=-1), 1.0
        )
