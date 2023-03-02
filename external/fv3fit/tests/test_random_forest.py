import fv3fit
from fv3fit.tfdataset import tfdataset_from_batches
import numpy as np
import xarray as xr
import pytest

n_features = 10


def get_batches(outputs):
    array = xr.DataArray(
        np.reshape(np.arange(30), (3, n_features)), dims=["sample", "z"]
    )
    ds = xr.Dataset({key: array for key in outputs})
    ds = ds.merge(xr.Dataset({"air_temperature": array}))
    return [ds, ds]


@pytest.mark.parametrize("outputs", (["Q1"], ["Q1", "Q2"]))
def test_random_forest_predict_dim_size(outputs):
    parameters = fv3fit.RandomForestHyperparameters(["air_temperature"], outputs)
    model = fv3fit.sklearn._random_forest.RandomForest(
        ["air_temperature"], outputs, parameters
    )
    batches = get_batches(outputs)
    model.fit(tfdataset_from_batches(batches))
    prediction = model.predict(batches[0][["air_temperature"]])
    assert prediction.sizes["z"] == n_features


@pytest.fixture(params=[1, 3])
def fitted_random_forest(request):
    n_estimators = request.param
    outputs = ["dQ1"]
    parameters = fv3fit.RandomForestHyperparameters(
        ["air_temperature"], outputs, n_estimators=n_estimators, n_jobs=1
    )
    model = fv3fit.sklearn._random_forest.RandomForest(
        ["air_temperature"], outputs, parameters
    )
    batches = get_batches(outputs)
    model.fit(tfdataset_from_batches(batches))
    return model, n_estimators


def test_rf_feature_importances(fitted_random_forest):
    rf, n_estimators = fitted_random_forest
    assert rf._feature_importances().shape == (n_estimators, n_features)
    assert rf._mean_feature_importances().shape == (n_features,)
    assert rf._std_feature_importances().shape == (n_features,)
