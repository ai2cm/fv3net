from typing import Iterable, Sequence
import xarray as xr
import pytest
import logging
from loaders import batches
from fv3net import regression
import fv3net.regression.keras as fv3keras
from fv3net.regression import shared
import numpy as np
import tempfile
import yaml
import subprocess
import os


logger = logging.getLogger(__name__)


@pytest.fixture(params=["DenseModel"])
def model_type(request) -> str:
    return request.param


@pytest.fixture
def hyperparameters(model_type) -> dict:
    if model_type == "DenseModel":
        return {"width": 8, "depth": 3}
    else:
        raise NotImplementedError(model_type)


@pytest.fixture
def input_variables() -> Iterable[str]:
    return ["air_temperature", "specific_humidity"]


@pytest.fixture
def output_variables() -> Iterable[str]:
    return ["dQ1", "dQ2"]


@pytest.fixture()
def batch_function(model_type: str) -> str:
    return "batches_from_geodata"


@pytest.fixture()
def batch_kwargs(data_source_name: str) -> dict:
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
    model_type: str,
    hyperparameters: dict,
    input_variables: Iterable[str],
    output_variables: Iterable[str],
    batch_function: str,
    batch_kwargs: dict,
) -> shared.ModelTrainingConfig:
    return shared.ModelTrainingConfig(
        model_type=model_type,
        hyperparameters=hyperparameters,
        input_variables=input_variables,
        output_variables=output_variables,
        batch_function=batch_function,
        batch_kwargs=batch_kwargs,
    )


@pytest.fixture
def train_config_filename(
    model_type: str,
    hyperparameters: dict,
    input_variables: Iterable[str],
    output_variables: Iterable[str],
    batch_function: str,
    batch_kwargs: dict,
) -> str:
    with tempfile.NamedTemporaryFile(mode="w") as f:
        yaml.dump(
            {
                "model_type": model_type,
                "hyperparameters": hyperparameters,
                "input_variables": input_variables,
                "output_variables": output_variables,
                "batch_function": batch_function,
                "batch_kwargs": batch_kwargs,
            },
            f,
        )
        yield f.name


@pytest.fixture
def training_batches(
    data_source_name: str,
    data_source_path: str,
    train_config: shared.ModelTrainingConfig,
) -> Sequence[xr.Dataset]:

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
def model(
    model_type: str,
    input_variables: Iterable[str],
    output_variables: Iterable[str],
    hyperparameters: dict,
) -> fv3keras.Model:
    return fv3keras.get_model(
        model_type, input_variables, output_variables, **hyperparameters
    )


@pytest.mark.regression
def test_training(
    model: fv3keras.Model,
    training_batches: Sequence[xr.Dataset],
    output_variables: Iterable[str],
):
    model.fit(training_batches)
    batch_dataset = training_batches[0]
    result = model.predict(batch_dataset)
    validate_dataset_result(result, batch_dataset, output_variables)


def validate_dataset_result(
    result: xr.Dataset, batch_dataset: xr.Dataset, output_variables: Iterable[str]
):
    """
    Use assertions to test whether the predicted output dataset metadata matches
    metadata from a reference, for the given variable names. Also checks output values
    are present.
    """
    missing_names = set(output_variables).difference(result.data_vars.keys())
    assert len(missing_names) == 0
    for varname in output_variables:
        assert result[varname].shape == batch_dataset[varname].shape, varname
        assert np.sum(np.isnan(result[varname].values)) == 0


@pytest.mark.regression
def test_dump_and_load_maintains_prediction(
    model: fv3keras.Model,
    training_batches: Sequence[xr.Dataset],
    output_variables: Iterable[str],
):
    model.fit(training_batches)
    with tempfile.TemporaryDirectory() as tmpdir:
        model.dump(tmpdir)
        loaded_model = model.__class__.load(tmpdir)
    batch_dataset = training_batches[0]
    loaded_result = loaded_model.predict(batch_dataset)
    validate_dataset_result(loaded_result, batch_dataset, output_variables)
    original_result = model.predict(batch_dataset)
    xr.testing.assert_equal(loaded_result, original_result)


@pytest.mark.regression
def test_training_integration(
    data_source_path: str,
    train_config_filename: str,
    tmp_path: str,
    data_source_name: str,
):
    """
    Test the bash endpoint for training the model produces the expected output files.
    """
    if data_source_name == "fine_res_apparent_sources":
        pytest.xfail("cannot test fine_res on disk until synth produces netcdf files")
    subprocess.check_call(
        [
            "python",
            "-m",
            "fv3net.regression.keras",
            data_source_path,
            train_config_filename,
            tmp_path,
            "--no-train-subdir-append",
        ]
    )
    required_names = ["model_data", "training_config.yml"]
    missing_names = set(required_names).difference(os.listdir(tmp_path))
    assert len(missing_names) == 0
