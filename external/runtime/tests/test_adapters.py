from runtime.adapters import RenamingAdapter
import xarray as xr
import numpy as np

import pytest


class Mock:
    input_vars_ = ["renamed_input"]

    def predict(self, x, arg2):
        in_ = x["renamed_input"]
        return xr.Dataset({"rename_output": in_})


def test_RenamingAdapter_predict():

    ds = xr.Dataset({"x": (["dim_0", "dim_1"], np.ones((5, 10)))})

    model = RenamingAdapter(Mock(), {"x": "renamed_input", "y": "rename_output"})
    out = model.predict(ds, None)
    assert "y" in out.data_vars


@pytest.mark.parametrize(
    "rename_dims, expected",
    [
        ({"dim_0": "xx"}, ["xx", "dim_1"]),
        ({"dim_1": "yy"}, ["dim_0", "yy"]),
        ({"dim_0": "xx", "dim_1": "yy"}, ["xx", "yy"]),
    ],
)
def test_RenamingAdapter_predict_orig_dims(rename_dims, expected):

    ds = xr.Dataset({"x": (["dim_0", "dim_1"], np.ones((5, 10)))})

    rename_dict = {"x": "renamed_input", "y": "rename_output"}
    rename_dict.update(rename_dims)
    model = RenamingAdapter(Mock(), rename_dict)
    out = model.predict(ds, None)
    assert list(out["y"].dims) == ["dim_0", "dim_1"]


def test_RenamingAdapter_variables():
    model = RenamingAdapter(Mock(), {"x": "renamed_input", "y": "rename_output"})
    assert model.variables == {"x"}
