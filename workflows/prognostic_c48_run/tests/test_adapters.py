from runtime.steppers.machine_learning import RenamingAdapter, MultiModelAdapter
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

    def __init__(self, input_variables=None, output_variables=None, output_scaling=1.0):
        self.input_variables = input_variables or ["renamed_input"]
        self.output_variables = output_variables or ["rename_output"]
        self.output_scaling = output_scaling

    def predict_columnwise(self, x, sample_dims=None):
        in_ = x[self.input_variables[0]] * self.output_scaling
        return xr.Dataset({self.output_variables[0]: in_})


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


def test_MultiModelAdapter_combines_predictions():
    ds = xr.Dataset({"x": (["dim_0", "dim_1"], np.ones((5, 10)))})
    model0 = MockPredictor(output_variables=["y0"], input_variables=["x"])
    model1 = MockPredictor(output_variables=["y1"], input_variables=["x"])
    combined_model = MultiModelAdapter([model0, model1])
    out = combined_model.predict_columnwise(ds)
    assert "y0" in out.data_vars and "y1" in out.data_vars


def test_MultiModelAdapter_exception_on_output_overlap():
    ds = xr.Dataset({"x": (["dim_0", "dim_1"], np.ones((5, 10)))})
    model0 = MockPredictor(output_variables=["y0"], input_variables=["x"])
    model1 = MockPredictor(
        output_variables=["y0"], input_variables=["x"], output_scaling=2.0
    )
    combined_model = MultiModelAdapter([model0, model1])
    with pytest.raises(xr.MergeError):
        combined_model.predict_columnwise(ds)
