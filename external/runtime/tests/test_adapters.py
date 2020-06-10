from runtime.adapters import RenamingAdapter
import xarray as xr
import numpy as np

import pytest


class Mock:
    """A predictor that expects these inputs::

        ["renamed_inputs"]
    
    and predicts these outputs::

        ["rename_output"]

    RenamingAdapter should be able to rename the input/output variables and
    dims to make this object work.
    """

    input_vars_ = ["renamed_input"]
    output_vars_ = ["rename_output"]

    def predict(self, x, arg2):
        in_ = x["renamed_input"]
        return xr.Dataset({"rename_output": in_})


def test_RenamingAdapter_predict_inputs_and_outputs_renamed():

    ds = xr.Dataset({"x": (["dim_0", "dim_1"], np.ones((5, 10)))})

    model = RenamingAdapter(Mock(), {"x": "renamed_input", "y": "rename_output"})
    out = model.predict(ds, None)
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
    m = Mock()
    ds = xr.Dataset({m.input_vars_[0]: (original_dims, np.ones((5, 10)))})
    model = RenamingAdapter(m, rename_dims)
    out = model.predict(ds, None)
    output_array = out[m.output_vars_[0]]
    assert list(output_array.dims) == expected


def test_RenamingAdapter_variables():
    model = RenamingAdapter(Mock(), {"x": "renamed_input", "y": "rename_output"})
    assert model.variables == {"x"}
