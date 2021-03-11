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


def _rad_model_dataset() -> xr.Dataset:

    arr = np.zeros((1,))
    dims = [
        "sample",
    ]

    data = xr.Dataset(
        {
            "specific_humidity": (dims, arr),
            "air_temperature": (dims, arr),
            "downward_shortwave": (dims, arr),
            "net_shortwave": (dims, arr),
            "downward_longwave": (dims, arr),
        }
    )

    return data


def get_mock_sklearn_model() -> fv3fit.Predictor:

    data = _model_dataset()

    nz = data.sizes["z"]
    heating_constant_K_per_s = np.zeros(nz)
    # include nonzero moistening to test for mass conservation
    moistening_constant_per_s = -np.full(nz, 1e-4 / 86400)
    wind_tendency_constant_m_per_s_per_s = np.full(nz, 1 / 86400)
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


def get_mock_rad_flux_model() -> fv3fit.Predictor:

    data = _rad_model_dataset()

    n_sample = data.sizes["sample"]
    downward_shortwave = np.full(n_sample, 300.0)
    net_shortwave = np.full(n_sample, 250.0)
    downward_longwave = np.full(n_sample, 400.0)
    constant = np.concatenate(
        [[downward_shortwave], [net_shortwave], [downward_longwave]]
    )
    estimator = RegressorEnsemble(
        DummyRegressor(strategy="constant", constant=constant)
    )

    model = SklearnWrapper(
        "sample",
        ["air_temperature", "specific_humidity"],
        ["downward_shortwave", "net_shortwave", "downward_longwave"],
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
