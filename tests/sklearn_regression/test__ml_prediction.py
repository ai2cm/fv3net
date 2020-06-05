from vcm import safe
import xarray as xr
import pytest

from loaders import SAMPLE_DIM_NAME
from loaders.mappers._ml_prediction import SklearnPredictionMapper


def mock_predict_function(feature_data_arrays):
    return sum(feature_data_arrays)


class MockSklearnWrappedModel:
    def __init__(self, input_vars, output_vars):
        self.input_vars = input_vars
        self.output_vars = output_vars

    def predict(self, ds_stacked):
        ds_pred = xr.Dataset()
        for output_var in self.output_vars:
            feature_vars = [ds_stacked[var] for var in self.input_vars]
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
        [[
            [[(100 * z) + (10 * y) + x for x in range(xdim)] for y in range(ydim)]
            for z in range(zdim)
        ]],
        dims=["initial_time", "z", "y", "x"],
        coords=coords,
    )
    ds = xr.Dataset({f"feature{i}": (i+1) * var for i in range(5)})
    return ds


@pytest.fixture
def base_mapper(gridded_dataset):
    return MockBaseMapper(gridded_dataset)


@pytest.fixture
def mock_model(request):
    input_vars, output_vars = request.params
    return MockSklearnWrappedModel(input_vars, output_vars)



 def test_ml_predict_wrapper(mock_model, base_mapper, gridded_dataset):
    mapper = SklearnPredictionMapper(
            base_mapper,
            mock_model,
            init_time_dim="initial_time",
            z_dim="z"
    )
    target = mock_predict_function(safe.get_variables(base_mapper[key], mock_model.input_vars))
    for key in mapper.keys():
        for output_var in mock_model.output_vars:
            assert mapper[output_var] == pytest.approx(target)