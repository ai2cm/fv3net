import xarray as xr
import numpy as np
import pytest

from offline_ml_diags._mapper import (
    PredictionMapper,
    PREDICT_COORD,
    TARGET_COORD,
    DERIVATION_DIM,
)

import fv3fit


class MockBaseMapper:
    def __init__(self, ds_template, n_keys=4):
        self._ds_template = ds_template
        self._keys = [f"2020050{i+1}.000000" for i in range(n_keys)]

    def __getitem__(self, key: str) -> xr.Dataset:
        ds = self._ds_template
        return ds

    def keys(self):
        return self._keys

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(self.keys())


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
def base_mapper():
    return MockBaseMapper(get_gridded_dataset(seed=0))


@pytest.fixture
def input_variables(request):
    return request.param


@pytest.fixture
def output_variables(request):
    return request.param


class MockPredictor(fv3fit.Predictor):
    """
    An object that returns a constant dataset for predictions.
    """

    def __init__(self, input_variables, output_variables, output_ds: xr.Dataset):
        self.input_variables = input_variables
        self.output_variables = output_variables
        non_predicted_names = set(output_ds.data_vars.keys()).difference(
            output_variables
        )
        self.output_ds = output_ds.drop(labels=non_predicted_names)
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


@pytest.mark.parametrize(
    "input_variables, output_variables",
    [
        pytest.param(["feature0", "feature1"], ["pred0"], id="keras_2_1"),
        pytest.param(["feature0", "feature1"], ["pred0", "pred1"], id="keras_2_2"),
        pytest.param(["feature0"], ["pred0", "pred1"], id="keras_1_2"),
        pytest.param(["feature0", "feature1"], ["pred0"], id="sklearn_2_1"),
        pytest.param(["feature0", "feature1"], ["pred0", "pred1"], id="sklearn_2_2"),
        pytest.param(["feature0"], ["pred0", "pred1"], id="sklearn_1_2"),
    ],
)
def test_ml_predict_mapper_insert_prediction(
    input_variables, output_variables, base_mapper
):
    mock_model = get_mock_model(input_variables, output_variables)
    variables = mock_model.output_variables + mock_model.input_variables
    mapper = PredictionMapper(
        base_mapper, mock_model, variables, z_dim="z", grid=xr.Dataset()
    )
    for key in mapper.keys():
        mapper_output = mapper[key]
        for var in mock_model.output_variables:
            assert set(mapper_output[var][DERIVATION_DIM].values) == {
                TARGET_COORD,
                PREDICT_COORD,
            }


@pytest.mark.parametrize(
    "input_variables, output_variables",
    [
        pytest.param(["feature0", "feature1"], ["pred0"], id="keras_2_1"),
        pytest.param(["feature0", "feature1"], ["pred0", "pred1"], id="keras_2_2"),
        pytest.param(["feature0", "feature1"], ["pred0"], id="sklearn_2_1"),
        pytest.param(["feature0", "feature1"], ["pred0", "pred1"], id="sklearn_2_2"),
    ],
)
def test_ml_predict_mapper(input_variables, output_variables, base_mapper):
    mock_model = get_mock_model(input_variables, output_variables)
    variables = mock_model.output_variables + mock_model.input_variables
    prediction_mapper = PredictionMapper(
        base_mapper, mock_model, variables, z_dim="z", grid=xr.Dataset()
    )
    for key in prediction_mapper.keys():
        prediction_output = prediction_mapper[key]
        for var in mock_model.output_variables:
            target = base_mapper[key][var]
            truth = (
                prediction_output[var]
                .sel({DERIVATION_DIM: "target"})
                .drop([DERIVATION_DIM, "time"])
            )
            prediction = (
                prediction_output[var]
                .sel({DERIVATION_DIM: "predict"})
                .drop([DERIVATION_DIM, "time"])
            )
            xr.testing.assert_allclose(truth, target)
            xr.testing.assert_allclose(prediction, mock_model.output_ds[var])


def test_prediction_mapper_inserts_grid_to_input(base_mapper):
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
    prediction_mapper = PredictionMapper(
        base_mapper, mock_model, variables, z_dim="z", grid=grid_ds
    )
    # need to call using any arbitrary key
    key = list(prediction_mapper.keys())[0]
    prediction_mapper[key]
    assert len(mock_model.call_datasets) == 1, "should be called only once"
    passed_ds = mock_model.call_datasets[0]
    for varname in grid_ds.data_vars:
        xr.testing.assert_identical(grid_ds[varname], passed_ds[varname].drop("time"))


def test_prediction_mapper_output_contains_input_value(base_mapper):
    input_variables, output_variables = ["feature0", "feature1"], ["pred0"]
    mock_model = get_mock_model(input_variables, output_variables)
    variables = mock_model.output_variables + mock_model.input_variables + ["pred1"]
    prediction_mapper = PredictionMapper(
        base_mapper, mock_model, variables, z_dim="z", grid=xr.Dataset()
    )
    # need to call using any arbitrary key
    key = list(prediction_mapper.keys())[0]
    output = prediction_mapper[key]
    assert "pred1" in output
    xr.testing.assert_equal(output["pred1"].drop("time"), base_mapper[key]["pred1"])


def test_prediction_mapper_output_contains_grid_value(base_mapper):
    input_variables, output_variables = ["feature0", "feature1"], ["pred0"]
    mock_model = get_mock_model(input_variables, output_variables)
    grid_ds = xr.Dataset(
        data_vars={
            "grid_var": xr.DataArray(
                np.random.randn(3, 4, 5), dims=["dim1", "dim2", "dim3"]
            )
        }
    )
    variables = mock_model.output_variables + mock_model.input_variables + ["grid_var"]
    prediction_mapper = PredictionMapper(
        base_mapper, mock_model, variables, z_dim="z", grid=grid_ds
    )
    # need to call using any arbitrary key
    key = list(prediction_mapper.keys())[0]
    output = prediction_mapper[key]
    assert "grid_var" in output
    xr.testing.assert_equal(output["grid_var"].drop("time"), grid_ds["grid_var"])
