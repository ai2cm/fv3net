import numpy as np
import xarray as xr
import pytest
from fv3fit._shared.models import SquashedOutputModel
from fv3fit._shared.config import SquashedOutputConfig
from fv3fit.testing import ConstantOutputPredictor


ARRAY = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]])
OUTPUT_DICT = {"a": ARRAY, "b": 0.1 * ARRAY}


def test_squashed_output_model_validate_name_in_outputs():
    squashing_configs = [
        SquashedOutputConfig(squash_by_name="a", squash_threshold=0.08)
    ]
    with pytest.raises(ValueError):
        SquashedOutputModel._validate(squashing_configs, output_variables=["b"])


def test_squashed_output_model_validate_repeat_squash_by():
    squashing_configs = [
        SquashedOutputConfig(squash_by_name="a", squash_threshold=0.08),
        SquashedOutputConfig(squash_by_name="a", squash_threshold=0.02),
    ]
    with pytest.raises(ValueError):
        SquashedOutputModel._validate(squashing_configs, output_variables=["a"])


def test_squashed_output_model_validate_repeat_targets():
    squashing_configs = [
        SquashedOutputConfig(
            squash_by_name="a",
            squash_threshold=0.02,
            additional_squash_target_names=["c"],
        ),
        SquashedOutputConfig(
            squash_by_name="b",
            squash_threshold=0.02,
            additional_squash_target_names=["c"],
        ),
    ]
    with pytest.raises(ValueError):
        SquashedOutputModel._validate(
            squashing_configs, output_variables=["a", "b", "c"]
        )


def test_squashed_output_model_validate_squash_by_in_targets():
    squashing_configs = [
        SquashedOutputConfig(
            squash_by_name="a",
            squash_threshold=0.02,
            additional_squash_target_names=["b"],
        ),
        SquashedOutputConfig(squash_by_name="b", squash_threshold=0.02),
    ]
    with pytest.raises(ValueError):
        SquashedOutputModel._validate(squashing_configs, output_variables=["a", "b"])


@pytest.mark.parametrize(
    ["additional_targets", "squash_to", "squash_threshold", "expected"],
    [
        pytest.param(
            ["b"],
            0.0,
            1.5,
            {"a": [[0.0, 0.0, 0.0, 0.0, 2.0]], "b": [[0.0, 0.0, 0.0, 0.0, 0.2]]},
            id="both_1.5_to_zero",
        ),
        pytest.param(
            [],
            0.0,
            1.5,
            {"a": [[0.0, 0.0, 0.0, 0.0, 2.0]], "b": OUTPUT_DICT["b"]},
            id="a_only_1.5_to_zero",
        ),
        pytest.param(
            ["b"],
            0.0,
            -1.5,
            {"a": [[0.0, -1.0, 0.0, 1.0, 2.0]], "b": [[0.0, -0.1, 0.0, 0.1, 0.2]]},
            id="both_-1.5_to_zero",
        ),
        pytest.param(
            ["b"],
            0.1,
            1.5,
            {"a": [[0.1, 0.1, 0.1, 0.1, 2.0]], "b": [[0.1, 0.1, 0.1, 0.1, 0.2]]},
            id="both_1.5_to_0.1",
        ),
    ],
)
def test_squashed_output_model_predict(
    additional_targets, squash_to, squash_threshold, expected
):
    base_model = ConstantOutputPredictor(
        input_variables=["n"], output_variables=["a", "b"]
    )
    base_model.set_outputs(
        **{k: v.squeeze() for k, v in OUTPUT_DICT.items()}
    )  # set_outputs adds the sample dim
    squashing_configs = [
        SquashedOutputConfig(
            squash_by_name="a",
            additional_squash_target_names=additional_targets,
            squash_threshold=squash_threshold,
            squash_to=squash_to,
        )
    ]
    squashed_model = SquashedOutputModel(base_model, squashing_configs)
    inputs = xr.Dataset({"n": xr.DataArray(ARRAY, dims=["x", "z"])})
    predictions = squashed_model.predict(inputs)
    for name in predictions:
        np.testing.assert_allclose(predictions[name].values, expected[name])
