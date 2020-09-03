from fv3fit import Predictor
import xarray as xr
import numpy as np

import pytest


class IdentityPredictor2D(Predictor):

    def predict(self, X):
        for variable in X:
            assert X[variable].ndim <= 2
        return X

    def load(self, *args, **kwargs):
        pass


def test__Predictor_predict_columnwise_dims_same_order():
    model = IdentityPredictor2D("sample", ["a"], ["a"])
    X = xr.Dataset({
        "a": (["x", "y", "z"], np.ones((3 ,4, 5)))
    })
    ans = model.predict_columnwise(X, sample_dims=["x", "y"])
    assert ans.a.dims == ("x" , "y", "z")


def test__Predictor_predict_columnwise_dims_same_order_2d_output():
    model = IdentityPredictor2D("sample", ["a"], ["b"])
    X = xr.Dataset({
        "a": (["x", "y", "z"], np.ones((3 ,4, 5))),
        "b": (["x", "y"], np.ones((3 ,4)))
    })
    ans = model.predict_columnwise(X, sample_dims=["x", "y"])
    assert ans.b.dims == ("x" , "y")

nx, ny, nz= 3, 4, 5
@pytest.mark.parametrize('coords', [
    {},
    {"x": np.arange(nx)},
    {"x": np.arange(nx), "y": np.arange(ny), "z": np.arange(nz)},

])
def test__Predictor_predict_columnwise_coordinates_same(coords,):
    model = IdentityPredictor2D("sample", ["a"], ["a"])
    X = xr.Dataset({
        "a": (["x", "y", "z"], np.ones((nx, ny, nz)))
    }, coords=coords)
    ans = model.predict_columnwise(X, sample_dims=["x", "y"])
    for coord in ans.coords:
        xr.testing.assert_equal(ans.coords[coord], X.coords[coord])