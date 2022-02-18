import numpy as np
import pytest
from fv3fit.keras._models.shared.output_range import OutputRange


@pytest.mark.parametrize(
    "min, max, expected",
    [
        pytest.param(None, None, [-2.0, -1.0, 0.0, 1.0, 2.0], id="no_bounds"),
        pytest.param(None, 1.5, [-2.0, -1.0, 0.0, 1.0, 1.5], id="upper_bound"),
        pytest.param(None, 1.0, [-2.0, -1.0, 0.0, 1.0, 1.0], id="upper_bound_equal"),
        pytest.param(-1.5, None, [-1.5, -1.0, 0.0, 1.0, 2.0], id="lower_bound"),
        pytest.param(-1.0, None, [-1.0, -1.0, 0.0, 1.0, 2.0], id="lower_bound_equal"),
        pytest.param(
            -1.5, 1.5, [-1.5, -1.0, 0.0, 1.0, 1.5], id="upper_and_lower_bounds"
        ),
    ],
)
def test_OutputRange(min, max, expected):
    range = OutputRange(min=min, max=max)
    output = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    limited_output = range.limit_output(output)
    assert np.array_equal(expected, limited_output)
