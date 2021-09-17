import xarray as xr
import numpy as np
import pytest

from fv3net.diagnostics.offline_ml_diags.compute_diags import (
    PREDICT_COORD,
    TARGET_COORD,
    DERIVATION_DIM_NAME,
    _get_predict_function,
)

import fv3fit


def get_gridded_dataset(seed=0):
    random = np.random.RandomState(seed=seed)
    nz, ny, nx = 2, 10, 10
    coords = {"z": range(nz), "y": range(ny), "x": range(nx)}
    # unique values for ease of set comparison in test
    var = xr.DataArray(random.randn(nz, ny, nx), dims=["z", "y", "x"], coords=coords,)
    ds = xr.Dataset(
        dict(
            **{f"feature{i}": (i + 1) + var for i in range(5)},
            **{f"pred{i}": -(i + 1) - var for i in range(5)},
        )
    )
    return ds


@pytest.fixture
def base_dataset():
    return get_gridded_dataset(seed=0)


class MockPredictor(fv3fit.Predictor):
    """
    An object that returns a constant dataset for predictions.
    """

    def __init__(self, input_variables, output_variables, output_ds: xr.Dataset):
        self.input_variables = input_variables
        self.output_variables = output_variables
        self.output_ds = output_ds
        self.call_datasets = []

    def predict(self, X: xr.Dataset):
        raise NotImplementedError()

    def predict_columnwise(self, X: xr.Dataset, *args, **kwargs) -> xr.Dataset:
        self.call_datasets.append(X)
        return self.output_ds

    def dump(self, path):
        raise NotImplementedError()

    @classmethod
    def load(cls, path):
        raise NotImplementedError()


def get_mock_model(input_variables, output_variables):
    # must use different random seed than for base_mapper fixture
    output_ds = get_gridded_dataset(seed=1)
    return MockPredictor(input_variables, output_variables, output_ds=output_ds)


CASES = [
    pytest.param(["feature0", "feature1"], ["pred0"], id="keras_2_1"),
    pytest.param(["feature0", "feature1"], ["pred0", "pred1"], id="keras_2_2"),
    pytest.param(["feature0"], ["pred0", "pred1"], id="keras_1_2"),
    pytest.param(["feature0", "feature1"], ["pred0"], id="sklearn_2_1"),
    pytest.param(["feature0", "feature1"], ["pred0", "pred1"], id="sklearn_2_2"),
    pytest.param(["feature0"], ["pred0", "pred1"], id="sklearn_1_2"),
]


@pytest.mark.parametrize(
    "input_variables, output_variables", CASES,
)
def test_predict_function_mapper_inserts_prediction_dim(
    input_variables, output_variables, base_dataset
):
    mock_model = get_mock_model(input_variables, output_variables)
    variables = mock_model.output_variables + mock_model.input_variables
    predict_function = _get_predict_function(mock_model, variables, grid=xr.Dataset())
    output = predict_function(base_dataset)
    for var in mock_model.output_variables:
        assert var in output
        assert set(output[var][DERIVATION_DIM_NAME].values) == {
            TARGET_COORD,
            PREDICT_COORD,
        }


@pytest.mark.parametrize("input_variables, output_variables", CASES)
def test_predict_function_inserts_prediction_values(
    input_variables, output_variables, base_dataset
):
    mock_model = get_mock_model(input_variables, output_variables)
    variables = mock_model.output_variables + mock_model.input_variables
    predict_function = _get_predict_function(mock_model, variables, grid=xr.Dataset())
    output = predict_function(base_dataset)
    for var in mock_model.output_variables:
        assert var in output
        print(output[var])
        target = base_dataset[var]
        truth = (
            output[var].sel({DERIVATION_DIM_NAME: "target"}).drop([DERIVATION_DIM_NAME])
        )
        prediction = (
            output[var]
            .sel({DERIVATION_DIM_NAME: "predict"})
            .drop([DERIVATION_DIM_NAME])
        )
        xr.testing.assert_allclose(truth, target)
        xr.testing.assert_allclose(prediction, mock_model.output_ds[var])


def test_predict_function_inserts_grid_before_calling_predict(base_dataset):
    input_variables, output_variables = ["feature0", "feature1"], ["pred0", "pred1"]
    mock_model = get_mock_model(input_variables, output_variables)
    grid_ds = xr.Dataset(
        data_vars={
            "grid_var": xr.DataArray(
                np.random.randn(3, 4, 5), dims=["dim1", "dim2", "dim3"]
            )
        }
    )
    variables = (
        mock_model.output_variables
        + mock_model.input_variables
        + list(grid_ds.data_vars.keys())
    )
    predict_function = _get_predict_function(mock_model, variables, grid=grid_ds)
    _ = predict_function(base_dataset)
    assert len(mock_model.call_datasets) == 1, "should be called only once"
    passed_ds = mock_model.call_datasets[0]
    for varname in grid_ds.data_vars:
        xr.testing.assert_identical(grid_ds[varname], passed_ds[varname])
