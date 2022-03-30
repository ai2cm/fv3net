from fv3fit import Predictor
import xarray as xr
import numpy as np

import pytest

from fv3fit._shared.predictor import DATASET_DIM_NAME


class IdentityPredictor2D(Predictor):
    def predict(self, X):
        return X[self.output_variables]

    def dump(self, *args, **kwargs):
        pass

    @classmethod
    def load(self, *args, **kwargs):
        pass


class InOutPredictor(Predictor):
    input_variables = ["in"]
    output_variables = ["out"]

    @classmethod
    def create(cls):
        return cls(cls.input_variables, cls.output_variables)

    def predict(self, x):
        return x.rename({"in": "out"})

    def dump(self, *args, **kwargs):
        pass

    @classmethod
    def load(self, *args, **kwargs):
        pass


@pytest.mark.parametrize("sample_dims", [("x", "y"), ("y", "x")])
def test__Predictor_predict_dims_same_order(sample_dims):
    model = IdentityPredictor2D(["a"], ["a"])
    X = xr.Dataset({"a": (["x", "y", "z"], np.ones((3, 4, 5)))})
    ans = model.predict(X)
    assert ans.a.dims == ("x", "y", "z")


@pytest.mark.parametrize("sample_dims", [("x", "y"), ("y", "x")])
def test__Predictor_predict_dims_same_order_InOutPredictor(sample_dims):
    model = InOutPredictor.create()
    shape = (3, 4, 5)
    ds = xr.Dataset({"in": (["z", "y", "x"], np.ones(shape))})
    output = model.predict(ds)
    assert output.out.dims == ("z", "y", "x")


def test__Predictor_predict_dims_same_order_2d_output():
    model = IdentityPredictor2D(["a", "b"], ["b"])
    X = xr.Dataset(
        {"a": (["x", "y", "z"], np.ones((3, 4, 5))), "b": (["x", "y"], np.ones((3, 4)))}
    )
    ans = model.predict(X)
    assert ans.b.dims == ("x", "y")


def test__Predictor_predict_dims_infers_feature_dim():
    model = IdentityPredictor2D(["a"], ["a"])
    X = xr.Dataset({"a": (["x", "y", "z"], np.ones((3, 4, 5)))})
    ans = model.predict(X)
    assert ans.a.dims == X.a.dims


nx, ny, nz = 3, 4, 5


@pytest.mark.parametrize(
    "coords",
    [
        {},
        {"x": np.arange(nx)},
        {"x": np.arange(nx), "y": np.arange(ny), "z": np.arange(nz)},
    ],
)
def test__Predictor_predict_coordinates_same(coords,):
    model = IdentityPredictor2D(["a"], ["a"])
    X = xr.Dataset({"a": (["x", "y", "z"], np.ones((nx, ny, nz)))}, coords=coords)
    ans = model.predict(X)
    for coord in ans.coords:
        xr.testing.assert_equal(ans.coords[coord], X.coords[coord])


def test__Predictor_predict_broadcast_dataset_dim_in_input():
    model = IdentityPredictor2D(["a", "b"], ["a"])
    X = xr.Dataset(
        {
            "a": (["x", "y", DATASET_DIM_NAME, "z"], np.ones((2, 3, 4, 5))),
            "b": (["x", "y", "z"], np.ones((2, 3, 5))),
        }
    )
    ans = model.predict(X)
    assert ans.a.dims == ("x", "y", DATASET_DIM_NAME, "z")
