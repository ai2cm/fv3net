import fv3fit
from fv3fit.tfdataset import tfdataset_from_batches
import numpy as np
import xarray as xr
import pytest


@pytest.mark.parametrize("outputs", (["Q1"], ["Q1", "Q2"]))
def test_random_forest_predict_dim_size(outputs):
    parameters = fv3fit.RandomForestHyperparameters(["air_temperature"], outputs)
    model = fv3fit.sklearn._random_forest.RandomForest(
        ["air_temperature"], outputs, parameters
    )
    array = xr.DataArray(np.reshape(np.arange(30), (3, 10)), dims=["sample", "z"])
    ds = xr.Dataset({key: array for key in outputs})
    ds = ds.merge(xr.Dataset({"air_temperature": array}))
    batches = [ds, ds]
    tfdataset = tfdataset_from_batches(batches)
    model.fit(tfdataset)
    prediction = model.predict(ds[["air_temperature"]])
    assert prediction.sizes["z"] == 10
