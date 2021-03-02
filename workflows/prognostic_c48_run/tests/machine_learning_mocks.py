import numpy as np
import xarray as xr
import tensorflow as tf
from toolz.functoolz import compose_left
from sklearn.dummy import DummyRegressor

import fv3fit
from fv3fit.sklearn import RegressorEnsemble, SklearnWrapper
from fv3fit.keras import DummyModel
from fv3fit.keras._models import AllPhysicsEmulator
from fv3fit._shared._transforms import stacking
from fv3fit._shared._transforms import emu_transforms

from runtime.loop import Emulator


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


def get_mock_keras_model() -> fv3fit.Predictor:

    input_variables = ["air_temperature", "specific_humidity"]
    output_variables = ["dQ1", "dQ2"]

    model = DummyModel("sample", input_variables, output_variables)
    model.fit([_model_dataset()])
    return model


def get_mock_all_physics_emulator() -> fv3fit.Predictor:

    ds = _model_dataset()
    input_variables = ["air_temperature", "specific_humidity"]
    output_variables = ["dQ1", "dQ2"]

    X_stacker = stacking.ArrayStacker.from_data(ds, input_variables)
    y_stacker = stacking.ArrayStacker.from_data(ds, output_variables)

    X_to_arr_func = compose_left(
        emu_transforms.extract_ds_arrays,
        X_stacker.stack,
    )

    model = _get_dummy_tensorflow_model(X.feature_size, y.feature_size)

    return AllPhysicsEmulator(
        "sample",
        input_variables,
        output_variables,
        model,
        X_to_arr_func,
        y_stacker.unstack,
    )


def _get_dummy_tensorflow_model(input_feature_size, output_feature_size):
    identity_matrix = tf.eye(input_feature_size, output_feature_size)

    input_ = tf.keras.layers.Input(input_feature_size)
    output_ = tf.keras.layers.Lambda(lambda x: x @ identity_matrix)(input_)
    model = tf.keras.Model(input_, output_)

    return model
