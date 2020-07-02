import xarray as xr
import pytest
import logging
from loaders import batches
from fv3net import regression
from fv3net.regression import shared
import numpy as np


logger = logging.getLogger(__name__)


@pytest.fixture
def model_type():
    return "DenseModel"


@pytest.fixture
def hyperparameters(model_type):
    if model_type == "DenseModel":
        return {"width": 8, "depth": 3}
    else:
        raise NotImplementedError(model_type)


@pytest.fixture
def input_variables():
    return ["air_temperature", "specific_humidity"]


@pytest.fixture
def output_variables():
    return ["dQ1", "dQ2"]


@pytest.fixture()
def batch_function(model_type):
    return "batches_from_geodata"


@pytest.fixture()
def batch_kwargs(data_source_name):
    if data_source_name == "one_step_tendencies":
        return {
            "timesteps_per_batch": 1,
            "init_time_dim_name": "initial_time",
            "mapping_function": "open_one_step",
            "timesteps": ["20160801.001500", "20160801.003000"],
        }
    elif data_source_name == "nudging_tendencies":
        return {
            "timesteps_per_batch": 1,
            "mapping_function": "open_merged_nudged",
            "timesteps": ["20160801.001500", "20160801.003000"],
            "mapping_kwargs": {
                "nudging_timescale_hr": 3,
                "i_start": 0,
                "rename_vars": {
                    "air_temperature_tendency_due_to_nudging": "dQ1",
                    "specific_humidity_tendency_due_to_nudging": "dQ2",
                },
            },
        }
    elif data_source_name == "fine_res_apparent_sources":
        return {
            "timesteps_per_batch": 1,
            "init_time_dim_name": "initial_time",
            "timesteps": ["20160801.001500", "20160801.003000"],
            "rename_variables": {},
        }


@pytest.fixture
def train_config(
    model_type,
    hyperparameters,
    input_variables,
    output_variables,
    batch_function,
    batch_kwargs,
):
    return shared.ModelTrainingConfig(
        model_type=model_type,
        hyperparameters=hyperparameters,
        input_variables=input_variables,
        output_variables=output_variables,
        batch_function=batch_function,
        batch_kwargs=batch_kwargs,
    )


@pytest.fixture
def training_batches(data_source_name, data_source_path, train_config):

    if data_source_name != "fine_res_apparent_sources":
        batched_data = regression.shared.load_data_sequence(
            data_source_path, train_config
        )
    else:
        # train.load_data_sequence is incompatible with synth's zarrs
        # (it looks for netCDFs); this is a patch until synth supports netCDF
        fine_res_ds = xr.open_zarr(data_source_path)
        mapper = {
            fine_res_ds.time.values[0]: fine_res_ds.isel(time=0),
            fine_res_ds.time.values[1]: fine_res_ds.isel(time=1),
        }
        batched_data = batches.batches_from_mapper(
            mapper,
            list(train_config.input_variables) + list(train_config.output_variables),
            **train_config.batch_kwargs,
        )
    return batched_data


@pytest.fixture
def model(model_type, input_variables, output_variables, hyperparameters):
    return regression.keras.get_model(
        model_type, input_variables, output_variables, **hyperparameters
    )


@pytest.mark.regression
def test_training(model, training_batches, output_variables):
    model.fit(training_batches)
    dataset = training_batches[0]
    result = model.predict(dataset)
    missing_names = set(output_variables).difference(result.data_vars.keys())
    assert len(missing_names) == 0
    for varname in output_variables:
        assert result[varname].shape == dataset[varname].shape, varname
        
        assert np.sum(np.isnan(result[varname].values)) == 0
