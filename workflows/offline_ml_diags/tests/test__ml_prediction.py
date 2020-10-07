import xarray as xr
import numpy as np
import pytest

from offline_ml_diags._mapper import (
    PredictionMapper,
    PREDICT_COORD,
    TARGET_COORD,
    DERIVATION_DIM,
)

from sklearn.dummy import DummyRegressor

from fv3fit.keras import DummyModel
from fv3fit.sklearn import SklearnWrapper
from vcm import safe


class MockBaseMapper:
    def __init__(self, ds_template):
        self._ds_template = ds_template
        self._keys = [f"2020050{i}.000000" for i in range(4)]

    def __getitem__(self, key: str) -> xr.Dataset:
        ds = self._ds_template
        return ds

    def keys(self):
        return self._keys

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(self.keys())


@pytest.fixture
def gridded_dataset():
    zdim, ydim, xdim = 2, 10, 10
    coords = {"z": range(zdim), "y": range(ydim), "x": range(xdim), "initial_time": [0]}
    # unique values for ease of set comparison in test
    var = xr.DataArray(
        [
            [
                [[(100 * z) + (10 * y) + x for x in range(xdim)] for y in range(ydim)]
                for z in range(zdim)
            ]
        ],
        dims=["initial_time", "z", "y", "x"],
        coords=coords,
    )
    ds = xr.Dataset(
        dict(
            **{f"feature{i}": (i + 1) * var for i in range(5)},
            **{f"pred{i}": (i + 1) * var for i in range(5)},
        )
    )
    return ds


@pytest.fixture
def base_mapper(gridded_dataset):
    return MockBaseMapper(gridded_dataset)


def mock_predict_function(sizes):
    return xr.DataArray(np.zeros(list(sizes.values())), dims=[dim for dim in sizes])


def get_mock_sklearn_model(input_variables, output_variables, ds):

    dummy = DummyRegressor(strategy="mean")
    model_wrapper = SklearnWrapper("sample", input_variables, output_variables, dummy)
    ds_stacked = safe.stack_once(
        ds, "sample", [dim for dim in ds.dims if dim != "z"]
    ).transpose("sample", "z")
    model_wrapper.fit(ds_stacked * 0)
    return model_wrapper


def get_mock_keras_model(input_variables, output_variables, ds):

    model = DummyModel("sample", input_variables, output_variables)

    ds_stacked = [
        safe.stack_once(ds, "sample", [dim for dim in ds.dims if dim != "z"]).transpose(
            "sample", "z"
        )
    ]

    model.fit(ds_stacked)

    return model


@pytest.fixture
def mock_model(request, gridded_dataset):
    model_type, input_variables, output_variables = request.param
    if model_type == "keras":
        return get_mock_keras_model(input_variables, output_variables, gridded_dataset)
    elif model_type == "sklearn":
        return get_mock_sklearn_model(
            input_variables, output_variables, gridded_dataset
        )
    else:
        raise ValueError("Invalid model type")


@pytest.mark.parametrize(
    "mock_model",
    [
        pytest.param(("keras", ["feature0", "feature1"], ["pred0"]), id="keras_2_1"),
        pytest.param(
            ("keras", ["feature0", "feature1"], ["pred0", "pred1"]), id="keras_2_2"
        ),
        pytest.param(("keras", ["feature0"], ["pred0", "pred1"]), id="keras_1_2"),
        pytest.param(
            ("sklearn", ["feature0", "feature1"], ["pred0"]), id="sklearn_2_1"
        ),
        pytest.param(
            ("sklearn", ["feature0", "feature1"], ["pred0", "pred1"]), id="sklearn_2_2"
        ),
        pytest.param(("sklearn", ["feature0"], ["pred0", "pred1"]), id="sklearn_1_2"),
    ],
    indirect=True,
)
def test_ml_predict_mapper_insert_prediction(mock_model, base_mapper, gridded_dataset):
    mapper = PredictionMapper(base_mapper, mock_model, z_dim="z",)
    for key in mapper.keys():
        mapper_output = mapper[key]
        for var in mock_model.output_variables:
            assert set(mapper_output[var][DERIVATION_DIM].values) == {
                TARGET_COORD,
                PREDICT_COORD,
            }


@pytest.mark.parametrize(
    "mock_model",
    [
        pytest.param(("keras", ["feature0", "feature1"], ["pred0"]), id="keras_2_1"),
        pytest.param(
            ("keras", ["feature0", "feature1"], ["pred0", "pred1"]), id="keras_2_2"
        ),
        pytest.param(
            ("sklearn", ["feature0", "feature1"], ["pred0"]), id="sklearn_2_1"
        ),
        pytest.param(
            ("sklearn", ["feature0", "feature1"], ["pred0", "pred1"]), id="sklearn_2_2"
        ),
    ],
    indirect=True,
)
def test_ml_predict_mapper(mock_model, base_mapper, gridded_dataset):
    mapper = PredictionMapper(base_mapper, mock_model, z_dim="z",)
    for key in mapper.keys():
        mapper_output = mapper[key]
        for var in mock_model.output_variables:
            target = base_mapper[key][var]
            truth = (
                mapper_output[var].sel({DERIVATION_DIM: "target"}).drop(DERIVATION_DIM)
            )
            prediction = (
                mapper_output[var].sel({DERIVATION_DIM: "predict"}).drop(DERIVATION_DIM)
            )

            xr.testing.assert_allclose(truth, target)
            # assume the model outputs 0.0
            xr.testing.assert_allclose(prediction, target * 0.0)
