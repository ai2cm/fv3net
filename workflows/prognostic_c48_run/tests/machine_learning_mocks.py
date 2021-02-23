import numpy as np
import xarray as xr
from sklearn.dummy import DummyRegressor

import fv3fit
from fv3fit.sklearn import RegressorEnsemble, SklearnWrapper
from fv3fit.keras import DummyModel


def _model_dataset() -> xr.Dataset:

    nz = 63
    arr = np.zeros((1, nz))
    dims = ["sample", "z"]

    data = xr.Dataset(
        {
            "specific_humidity": (dims, arr),
            "air_temperature": (dims, arr),
            "dQ1": (dims, arr),
            "dQ2": (dims, arr),
            "dQu": (dims, arr),
            "dQv": (dims, arr),
        }
    )

    return data

def get_mock_sklearn_model() -> fv3fit.Predictor:

    data = _model_dataset()

    nz = data.sizes["z"]
    heating_constant_K_per_s = np.zeros(nz)
    # include nonzero moistening to test for mass conservation
    moistening_constant_per_s = -np.full(nz, 1e-4 / 86400)
    wind_tendency_constant_m_per_s_per_s = np.zeros(nz)
    constant = np.concatenate(
        [
            heating_constant_K_per_s,
            moistening_constant_per_s,
            wind_tendency_constant_m_per_s_per_s,
            wind_tendency_constant_m_per_s_per_s,
        ]
    )
    estimator = RegressorEnsemble(
        DummyRegressor(strategy="constant", constant=constant)
    )

    model = SklearnWrapper(
        "sample",
        ["specific_humidity", "air_temperature"],
        ["dQ1", "dQ2", "dQu", "dQv"],
        estimator,
    )

    # needed to avoid sklearn.exceptions.NotFittedError
    model.fit([data])
    return model


def get_mock_keras_model() -> fv3fit.Predictor:

    input_variables = ["air_temperature", "specific_humidity"]
    output_variables = ["dQ1", "dQ2"]

    model = DummyModel("sample", input_variables, output_variables)
    model.fit([_model_dataset()])
    return model


