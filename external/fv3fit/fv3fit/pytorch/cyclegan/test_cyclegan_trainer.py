from fv3fit.pytorch.cyclegan.cyclegan_trainer import get_percentile
import numpy as np
import pytest


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
