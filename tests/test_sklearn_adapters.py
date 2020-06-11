from fv3net.regression.sklearn.adapters import RenamingAdapter, StackingAdapter
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

    input_vars_ = ["renamed_input"]
    output_vars_ = ["rename_output"]

    def predict(self, x):
        in_ = x["renamed_input"]
        return xr.Dataset({"rename_output": in_})


class MockSklearnWrapper:
    """A mock with the same interface as SklearnWrapper"""

    input_vars_ = ["in"]
    output_vars_ = ["out"]

    def predict(self, x, dim):
        assert x["in"].ndim == 2
        return x.rename({"in": "out"})


def test_RenamingAdapter_predict_inputs_and_outputs_renamed():

    ds = xr.Dataset({"x": (["dim_0", "dim_1"], np.ones((5, 10)))})

    model = RenamingAdapter(
        MockPredictor(), {"x": "renamed_input"}, {"y": "rename_output"}
    )
    out = model.predict(ds)
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
    ds = xr.Dataset({m.input_vars_[0]: (original_dims, np.ones((5, 10)))})
    model = RenamingAdapter(m, rename_dims)
    out = model.predict(ds)
    output_array = out[m.output_vars_[0]]
    assert list(output_array.dims) == expected


def test_RenamingAdapter_input_vars_():
    model = RenamingAdapter(MockPredictor(), {"x": "renamed_input"})
    assert model.input_vars_ == {"x"}


def test_StackingAdapter_input_vars_():
    model = MockSklearnWrapper()
    wrapper = StackingAdapter(model, sample_dims=())
    assert set(wrapper.input_vars_) == set(model.input_vars_)


def test_StackingAdapter_predict():
    model = MockSklearnWrapper()
    wrapper = StackingAdapter(model, sample_dims=["y", "x"])

    shape = (3, 4, 5)

    ds = xr.Dataset({"in": (["z", "y", "x"], np.ones(shape))})
    output = wrapper.predict(ds)
    assert set(output.out.dims) == {"z", "y", "x"}
    assert output.out.shape == shape
