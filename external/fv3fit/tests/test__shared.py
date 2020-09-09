from fv3fit import Predictor
import xarray as xr
import numpy as np

import pytest


class IdentityPredictor2D(Predictor):
    def predict(self, X):
        for variable in X:
            assert X[variable].ndim <= 2
        return X[self.output_variables]

    def load(self, *args, **kwargs):
        pass


class InOutPredictor(Predictor):
    input_variables = ["in"]
    output_variables = ["out"]

    @classmethod
    def create(cls):
        return cls("sample", cls.input_variables, cls.output_variables)

    def predict(self, x):
        assert x["in"].ndim == 2
        return x.rename({"in": "out"})

    def load(self, *args, **kwargs):
        pass


@pytest.mark.parametrize("sample_dims", [("x", "y"), ("y", "x")])
def test__Predictor_predict_columnwise_dims_same_order(sample_dims,):
    model = IdentityPredictor2D("sample", ["a"], ["a"])
    X = xr.Dataset({"a": (["x", "y", "z"], np.ones((3, 4, 5)))})
    ans = model.predict_columnwise(X, sample_dims=sample_dims)
    assert ans.a.dims == ("x", "y", "z")


@pytest.mark.parametrize("sample_dims", [("x", "y"), ("y", "x")])
def test__Predictor_predict_columnwise_dims_same_order_InOutPredictor(sample_dims):
    model = InOutPredictor.create()
    shape = (3, 4, 5)
    ds = xr.Dataset({"in": (["z", "y", "x"], np.ones(shape))})
    output = model.predict_columnwise(ds, sample_dims=sample_dims)
    assert output.out.dims == ("z", "y", "x")


def test__Predictor_predict_columnwise_dims_same_order_2d_output():
    model = IdentityPredictor2D("sample", ["a", "b"], ["b"])
    X = xr.Dataset(
        {"a": (["x", "y", "z"], np.ones((3, 4, 5))), "b": (["x", "y"], np.ones((3, 4)))}
    )
    ans = model.predict_columnwise(X, sample_dims=["x", "y"])
    assert ans.b.dims == ("x", "y")


def test__Predictor_predict_columnwise_dims_infers_feature_dim():
    model = IdentityPredictor2D("sample", ["a"], ["a"])
    X = xr.Dataset({"a": (["x", "y", "z"], np.ones((3, 4, 5)))})
    ans = model.predict_columnwise(X, feature_dim=["z"])
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
def test__Predictor_predict_columnwise_coordinates_same(coords,):
    model = IdentityPredictor2D("sample", ["a"], ["a"])
    X = xr.Dataset({"a": (["x", "y", "z"], np.ones((nx, ny, nz)))}, coords=coords)
    ans = model.predict_columnwise(X, sample_dims=["x", "y"])
    for coord in ans.coords:
        xr.testing.assert_equal(ans.coords[coord], X.coords[coord])
