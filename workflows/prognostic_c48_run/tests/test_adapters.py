from runtime import RenamingAdapter
import xarray as xr
import numpy as np
import pytest


class MockPredictor:
    """A predictor that expects these inputs::

        ["renamed_inputs"]
    
    and predicts these outputs::

        ["rename_output"]

    RenamingAdapter should be able to rename the input/output variables and
    dims to make this object work.
    """

    input_variables = ["renamed_input"]
    output_variables = ["rename_output"]

    def predict_columnwise(self, x, sample_dims=None):
        in_ = x["renamed_input"]
        return xr.Dataset({"rename_output": in_})


def test_RenamingAdapter_predict_inputs_and_outputs_renamed():

    ds = xr.Dataset({"x": (["dim_0", "dim_1"], np.ones((5, 10)))})

    model = RenamingAdapter(
        MockPredictor(), {"x": "renamed_input"}, {"y": "rename_output"}
    )
    out = model.predict_columnwise(ds)
    assert "y" in out.data_vars


@pytest.mark.parametrize(
    "original_dims, rename_dims, expected",
    [
        (["dim_0", "dim_1"], {"dim_0": "xx"}, ["xx", "dim_1"]),
        (["dim_0", "dim_1"], {"dim_1": "yy"}, ["dim_0", "yy"]),
        (["dim_0", "dim_1"], {"dim_0": "xx", "dim_1": "yy"}, ["xx", "yy"]),
    ],
)
def test_RenamingAdapter_predict_renames_dims_correctly(
    original_dims, rename_dims, expected
):
    m = MockPredictor()
    ds = xr.Dataset({m.input_variables[0]: (original_dims, np.ones((5, 10)))})
    model = RenamingAdapter(m, rename_dims)
    out = model.predict_columnwise(ds)
    output_array = out[m.output_variables[0]]
    assert list(output_array.dims) == expected


def test_RenamingAdapter_input_vars_():
    model = RenamingAdapter(MockPredictor(), {"x": "renamed_input"})
    assert model.input_variables == {"x"}
