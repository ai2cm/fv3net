import numpy as np
import pytest
from fv3net.diagnostics.prognostic_run.metrics import compute_percentile


@pytest.mark.parametrize(
    ["percentile", "expected_value"],
    [(0, 0.5), (5, 0.5), (10, 0.5), (20, 1.5), (45, 2.5), (99, 3.5)],
)
def test_compute_percentile(percentile, expected_value):
    bins = np.array([0, 1, 2, 3])
    bin_widths = np.array([1, 1, 1, 1])
    frequency = np.array([0.1, 0.1, 0.4, 0.4])
    value = compute_percentile(percentile, frequency, bins, bin_widths)
    assert value == expected_value


def test_compute_percentile_raise_value_error():
    bins = np.array([0, 1, 2, 3])
    bin_widths = np.array([1, 1, 1, 0.5])
    frequency = np.array([0.1, 0.1, 0.4, 0.4])
    with pytest.raises(ValueError):
        compute_percentile(0, frequency, bins, bin_widths)
