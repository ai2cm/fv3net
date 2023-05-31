import numpy as np
import xarray as xr
import pytest
from fv3fit._shared.models import SquashedOutputModel
from fv3fit.testing import ConstantOutputPredictor


ARRAY = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]])
OUTPUT_DICT = {"a": ARRAY, "b": 0.1 * ARRAY}


@pytest.mark.parametrize(
    ["squash_to", "squash_threshold", "expected"],
    [
        pytest.param(None, None, OUTPUT_DICT, id="no squash"),
        pytest.param(
            0.0,
            1.5,
            {"a": [[0.0, 0.0, 0.0, 0.0, 2.0]], "b": [[0.0, 0.0, 0.0, 0.0, 0.2]]},
            id="1.5_to_zero",
        ),
        pytest.param(
            0.0,
            -1.5,
            {"a": [[0.0, -1.0, 0.0, 1.0, 2.0]], "b": [[0.0, -0.1, 0.0, 0.1, 0.2]]},
            id="-1.5_to_zero",
        ),
    ],
)
def test_squashed_output_model_predict(squash_to, squash_threshold, expected):
    base_model = ConstantOutputPredictor(
        input_variables=["n"], output_variables=["a", "b"]
    )
    base_model.set_outputs(
        **{k: v.squeeze() for k, v in OUTPUT_DICT.items()}
    )  # set_outputs adds the sample dim
    squashed_model = SquashedOutputModel(
        base_model,
        squash_by_name=(None if squash_to is None else "a"),
        squash_threshold=squash_threshold,
        squash_to=squash_to,
    )
    inputs = xr.Dataset({"n": xr.DataArray(ARRAY, dims=["x", "z"])})
    predictions = squashed_model.predict(inputs)
    for name in predictions:
        np.testing.assert_allclose(predictions[name].values, expected[name])


def test_squashed_output_model_name_error():
    base_model = ConstantOutputPredictor(input_variables=["x"], output_variables=["a"])
    with pytest.raises(ValueError):
        SquashedOutputModel(
            base_model, squash_by_name="a", squash_threshold=None, squash_to=None
        )


def test_squashed_output_model_not_in_output_error():
    base_model = ConstantOutputPredictor(input_variables=["x"], output_variables=["b"])
    with pytest.raises(ValueError):
        SquashedOutputModel(
            base_model, squash_by_name="a", squash_threshold=0.5, squash_to=0.0
        )
