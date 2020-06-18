import xarray as xr
import pytest

from loaders import SAMPLE_DIM_NAME
from fv3net.regression.sklearn._mapper import (
    SklearnPredictionMapper,
    PREDICT_COORD,
    TARGET_COORD,
    DERIVATION_DIM,
)


def mock_predict_function(feature_data_arrays):
    return sum(feature_data_arrays)


class MockSklearnWrappedModel:
    def __init__(self, input_vars, output_vars):
        self.input_vars_ = input_vars
        self.output_vars_ = output_vars

    def predict(self, ds_stacked, sample_dim=SAMPLE_DIM_NAME):
        ds_pred = xr.Dataset()
        for output_var in self.output_vars_:
            feature_vars = [ds_stacked[var] for var in self.input_vars_]
            mock_prediction = mock_predict_function(feature_vars)
            ds_pred[output_var] = mock_prediction
        return ds_pred


class MockBaseMapper:
    def __init__(self, ds_template):
        self._ds_template = ds_template
        self._keys = [f"2020050{i}.000000" for i in range(4)]

    def __getitem__(self, key: str) -> xr.Dataset:
        ds = self._ds_template
        ds.coords["initial_time"] = [key]
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
    ds = xr.Dataset({f"feature{i}": (i + 1) * var for i in range(5)})
    return ds


@pytest.fixture
def base_mapper(gridded_dataset):
    return MockBaseMapper(gridded_dataset)


@pytest.fixture
def mock_model(request):
    input_vars, output_vars = request.param
    return MockSklearnWrappedModel(input_vars, output_vars)


@pytest.mark.parametrize(
    "mock_model",
    [
        (["feature0", "feature1"], ["pred0"]),
        (["feature0", "feature1"], ["pred0", "pred1"]),
    ],
    indirect=True,
)
def test_ml_predict_wrapper_insert_prediction(mock_model, base_mapper, gridded_dataset):
    mapper = SklearnPredictionMapper(
        base_mapper, mock_model, init_time_dim="initial_time", z_dim="z",
    )
    for key in mapper.keys():
        mapper_output = mapper[key]
        for var in mock_model.output_vars_:
            assert set(mapper_output[var][DERIVATION_DIM].values) == {
                TARGET_COORD,
                PREDICT_COORD,
            }


@pytest.mark.parametrize(
    "mock_model",
    [
        (["feature0", "feature1"], ["pred0"]),
        (["feature0", "feature1"], ["pred0", "pred1"]),
    ],
    indirect=True,
)
def test_ml_predict_wrapper(mock_model, base_mapper, gridded_dataset):
    mapper = SklearnPredictionMapper(
        base_mapper, mock_model, init_time_dim="initial_time", z_dim="z",
    )
    for key in mapper.keys():
        mapper_output = mapper[key]
        target = mock_predict_function(
            [base_mapper[key][var] for var in mock_model.input_vars_]
        )
        for var in mock_model.output_vars_:
            assert sum(
                (
                    mapper_output[var].sel({DERIVATION_DIM: PREDICT_COORD}) - target
                ).values
            ) == pytest.approx(0)


@pytest.mark.parametrize(
    "mock_model",
    [([], ["pred0"]), (["feature0", "feature10"], ["pred0"])],
    indirect=True,
)
def test_ml_predict_wrapper_invalid_usage(mock_model, base_mapper, gridded_dataset):
    mapper = SklearnPredictionMapper(
        base_mapper, mock_model, init_time_dim="initial_time", z_dim="z",
    )
    with pytest.raises(Exception):
        for key in mapper.keys():
            mapper[key]
