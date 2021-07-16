import dataclasses
from typing import Optional, Sequence
import fv3fit
import fv3fit.train
from fv3fit._shared.io import register
import yaml
import pytest
import os
import numpy as np
import cftime
import xarray as xr
import loaders
import vcm
from unittest import mock


@pytest.fixture
def mock_dataset():

    n_x, n_y, n_z, n_tile, n_time = (8, 8, 10, 6, 9)
    arr = np.zeros((n_time, n_tile, n_z, n_y, n_x))
    arr_surface = np.zeros((n_time, n_tile, n_y, n_x))
    dims = ["time", "tile", "z", "y", "x"]
    dims_surface = ["time", "tile", "y", "x"]

    data = xr.Dataset(
        {
            "specific_humidity": (dims, arr),
            "air_temperature": (dims, arr),
            "downward_shortwave": (dims_surface, arr_surface),
            "net_shortwave": (dims_surface, arr_surface),
            "downward_longwave": (dims_surface, arr_surface),
            "dQ1": (dims, arr),
            "dQ2": (dims, arr),
            "dQu": (dims, arr),
            "dQv": (dims, arr),
        },
        coords={
            "time": [
                cftime.DatetimeJulian(2016, 8, day) for day in range(1, 1 + n_time)
            ]
        },
    )

    return data


@dataclasses.dataclass
class MainArgs:
    data_path: str
    train_config: str
    train_data_config: str
    val_data_config: str
    output_path: str
    local_download_path: Optional[str] = None


class MockHyperparameters:
    param1: str = ""


@pytest.fixture
def mock_train_dense_model():
    train_mock = mock.MagicMock(name="train_dense_model")
    train_mock.return_value = mock.MagicMock(
        name="train_dense_model_return", spec=fv3fit.Predictor
    )
    register("mock")(train_mock.return_value.__class__)
    original_func = fv3fit.get_training_function("DenseModel")
    try:
        fv3fit._shared.config.register_training_function(
            "DenseModel", fv3fit.DenseHyperparameters
        )(train_mock)
        yield train_mock
    finally:
        fv3fit._shared.config.register_training_function(
            "DenseModel", fv3fit.DenseHyperparameters
        )(original_func)
        register._model_types.pop("mock")


@pytest.fixture
def mock_load_batches():
    magic_load_mock = mock.MagicMock(name="load_batches")
    with mock.patch.object(loaders.BatchesConfig, "load_batches", magic_load_mock):
        yield magic_load_mock


@pytest.mark.parametrize("additional_variables", [[], ["downward_shortwave"]])
def test_main(
    mock_dataset: xr.Dataset,
    tmpdir,
    mock_load_batches: mock.MagicMock,
    mock_train_dense_model: mock.MagicMock,
    additional_variables: Sequence[str],
):
    """
    Test of fv3fit.train main function only, using mocks for training function
    and data loading.
    """
    mock_load_batches.return_value = [mock_dataset for _ in range(6)]
    args, input_variables, output_variables, hyperparameters, output_path = get_config(
        tmpdir, mock_dataset, additional_variables, []
    )
    all_variables = input_variables + output_variables + additional_variables
    fv3fit.train.main(args)
    mock_load_batches.assert_called_with(variables=all_variables)
    assert mock_load_batches.call_args_list[0] == mock_load_batches.call_args_list[1]
    assert mock_load_batches.call_count == 2
    mock_train_dense_model.assert_called_once_with(
        input_variables=input_variables,
        output_variables=output_variables,
        hyperparameters=hyperparameters,
        train_batches=mock_load_batches.return_value,
        validation_batches=mock_load_batches.return_value,
    )
    mock_predictor = mock_train_dense_model.return_value
    mock_predictor.dump.assert_called_once()
    assert output_path in mock_predictor.dump.call_args[0]


@pytest.mark.parametrize("additional_variables", [[], ["downward_shortwave"]])
def test_main_with_derived_output_variables(
    mock_dataset: xr.Dataset,
    tmpdir,
    mock_load_batches: mock.MagicMock,
    mock_train_dense_model: mock.MagicMock,
    additional_variables: Sequence[str],
):
    """
    Test of fv3fit.train main function only, using mocks for training function
    and data loading.
    """
    mock_load_batches.return_value = [mock_dataset for _ in range(6)]
    derived_output_variables = ["downwelling_shortwave"]
    args, input_variables, output_variables, hyperparameters, output_path = get_config(
        tmpdir, mock_dataset, additional_variables, derived_output_variables
    )
    all_variables = input_variables + output_variables + additional_variables
    with mock.patch("fv3fit.DerivedModel") as MockDerivedModel:
        MockDerivedModel.return_value = mock.MagicMock(
            name="derived_model_return", spec=fv3fit.Predictor
        )
        fv3fit.train.main(args)
        MockDerivedModel.assert_called_once_with(
            mock_train_dense_model.return_value, derived_output_variables
        )
    mock_load_batches.assert_called_with(variables=all_variables)
    assert mock_load_batches.call_args_list[0] == mock_load_batches.call_args_list[1]
    assert mock_load_batches.call_count == 2
    mock_train_dense_model.assert_called_once_with(
        input_variables=input_variables,
        output_variables=output_variables,
        hyperparameters=hyperparameters,
        train_batches=mock_load_batches.return_value,
        validation_batches=mock_load_batches.return_value,
    )
    mock_predictor = MockDerivedModel.return_value
    # here we make sure the wrapped model is what's dumped
    mock_predictor.dump.assert_called_once()
    assert output_path in mock_predictor.dump.call_args[0]
    # this doesn't get called because we've mocked DerivedModel.dump
    assert mock_train_dense_model.return_value.dump.call_count == 0


def get_config(tmpdir, mock_dataset, additional_variables, derived_output_variables):
    model_type = "DenseModel"
    hyperparameters = fv3fit.DenseHyperparameters(depth=2, width=8)
    base_dir = str(tmpdir)
    input_variables = ["air_temperature", "specific_humidity"]
    output_variables = ["dQ1", "dQ2"]
    training_config = fv3fit.TrainingConfig(
        model_type=model_type,
        input_variables=input_variables,
        output_variables=output_variables,
        hyperparameters=hyperparameters,
        additional_variables=additional_variables,
        derived_output_variables=derived_output_variables,
    )
    assert len(mock_dataset["time"]) >= 9, "hard-coded assumption in test below"
    train_times = [vcm.encode_time(dt) for dt in mock_dataset["time"][:6].values]
    validation_times = [vcm.encode_time(dt) for dt in mock_dataset["time"][6:9].values]
    train_data_config = loaders.BatchesConfig(
        data_path=base_dir,
        batches_function="batches_from_geodata",
        batches_kwargs=dict(
            mapping_function="open_zarr",
            timesteps=train_times,
            needs_grid=False,
            res="c8_random_values",
            timesteps_per_batch=3,
        ),
    )
    validation_data_config = loaders.BatchesConfig(
        data_path=base_dir,
        batches_function="batches_from_geodata",
        batches_kwargs=dict(
            mapping_function="open_zarr",
            timesteps=validation_times,
            needs_grid=False,
            res="c8_random_values",
            timesteps_per_batch=3,
        ),
    )
    data_path = os.path.join(base_dir, "data")
    train_data_filename = os.path.join(base_dir, "train_data.yaml")
    validation_data_filename = os.path.join(base_dir, "validation_data.yaml")
    training_filename = os.path.join(base_dir, "training.yaml")
    with open(train_data_filename, "w") as f:
        yaml.dump(dataclasses.asdict(train_data_config), f)
    with open(validation_data_filename, "w") as f:
        yaml.dump(dataclasses.asdict(validation_data_config), f)
    with open(training_filename, "w") as f:
        yaml.dump(dataclasses.asdict(training_config), f)
    output_path = os.path.join(base_dir, "output")

    args = MainArgs(
        data_path=data_path,
        train_config=training_filename,
        train_data_config=train_data_filename,
        val_data_config=validation_data_filename,
        output_path=output_path,
    )
    return args, input_variables, output_variables, hyperparameters, output_path
