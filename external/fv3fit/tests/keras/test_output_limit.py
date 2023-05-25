import numpy as np
import pytest
from fv3fit.keras._models.shared.output_limit import OutputLimit, OutputSquashConfig


OUTPUT = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
OUTPUT_DICT = {"a": OUTPUT, "b": 0.1 * OUTPUT}


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
def test_OutputLimit(min, max, expected):
    range = OutputLimit(min=min, max=max)
    limited_output = range.limit_output(OUTPUT)
    assert np.array_equal(expected, limited_output)


@pytest.mark.parametrize(
    ["squash_to", "squash_threshold", "expected"],
    [
        pytest.param(None, None, OUTPUT_DICT, id="no squash"),
        pytest.param(
            0.0,
            1.5,
            {"a": [0.0, 0.0, 0.0, 0.0, 2.0], "b": [0.0, 0.0, 0.0, 0.0, 0.2]},
            id="1.5_to_zero",
        ),
        pytest.param(
            0.0,
            -1.5,
            {"a": [0.0, -1.0, 0.0, 1.0, 2.0], "b": [0.0, -0.1, 0.0, 0.1, 0.2]},
            id="-1.5_to_zero",
        ),
    ],
)
def test_OutputSquashConfig_squash_outputs(squash_to, squash_threshold, expected):
    squash_by_name = "a" if squash_to is not None else None
    squash = OutputSquashConfig(
        squash_to=squash_to,
        squash_threshold=squash_threshold,
        squash_by_name=squash_by_name,
    )
    squashed_output = squash.squash_outputs(
        names=tuple(OUTPUT_DICT.keys()), outputs=tuple(OUTPUT_DICT.values())
    )
    squashed_output_dict = {k: v for k, v in zip(OUTPUT_DICT.keys(), squashed_output)}
    for name in squashed_output_dict:
        np.testing.assert_allclose(squashed_output_dict[name], expected[name])


def test_OutputSquashConfig_squash_name_error():
    with pytest.raises(ValueError):
        OutputSquashConfig(squash_by_name="a")


def test_OutputSquashConfig_not_in_output_error():
    squash = OutputSquashConfig(squash_to=0.0, squash_threshold=0.5, squash_by_name="a")
    output_without_name = {"b": OUTPUT}
    with pytest.raises(ValueError):
        squash.squash_outputs(
            names=tuple(output_without_name.keys()),
            outputs=tuple(output_without_name.values()),
        )
