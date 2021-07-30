import dataclasses
import subprocess
from typing import Any, Optional, Sequence
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


def get_mock_dataset(n_time):

    n_x, n_y, n_z, n_tile = (8, 8, 10, 6)
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
    training_config: str
    training_data_config: str
    validation_data_config: str
    output_path: str
    local_download_path: Optional[str] = None


class MockHyperparameters:
    param1: str = ""


@dataclasses.dataclass
class TestConfig:
    args: MainArgs
    input_variables: Sequence[str]
    output_variables: Sequence[str]
    output_path: str
    mock_dataset: xr.Dataset


@dataclasses.dataclass
class CallArtifacts:
    output_path: str
    all_variables: Sequence[str]
    MockDerivedModel: mock.Mock
    input_variables: Sequence[str]
    output_variables: Sequence[str]
    hyperparameters: Any


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


def call_main(
    tmpdir, mock_load_batches, additional_variables, derived_output_variables,
):
    model_type = "DenseModel"
    hyperparameters = fv3fit.DenseHyperparameters(depth=2, width=8)
    config = get_config(
        tmpdir,
        additional_variables,
        derived_output_variables,
        model_type,
        hyperparameters,
    )
    mock_load_batches.return_value = [config.mock_dataset for _ in range(6)]
    all_variables = (
        config.input_variables + config.output_variables + additional_variables
    )
    with mock.patch("fv3fit.DerivedModel") as MockDerivedModel:
        MockDerivedModel.return_value = mock.MagicMock(
            name="derived_model_return", spec=fv3fit.Predictor
        )
        fv3fit.train.main(config.args)
    return CallArtifacts(
        config.output_path,
        all_variables,
        MockDerivedModel,
        config.input_variables,
        config.output_variables,
        hyperparameters,
    )


@pytest.mark.parametrize("additional_variables", [[], ["downward_shortwave"]])
@pytest.mark.parametrize("derived_output_variables", [[], ["downwelling_shortwave"]])
def test_main_calls_load_batches_correctly(
    tmpdir,
    mock_load_batches: mock.MagicMock,
    mock_train_dense_model: mock.MagicMock,
    additional_variables: Sequence[str],
    derived_output_variables: Sequence[str],
):
    """
    Test of fv3fit.train main function only, using mocks for training function
    and data loading.
    """
    artifacts = call_main(
        tmpdir, mock_load_batches, additional_variables, derived_output_variables,
    )
    mock_load_batches.assert_called_with(variables=artifacts.all_variables)
    assert mock_load_batches.call_args_list[0] == mock_load_batches.call_args_list[1]
    assert mock_load_batches.call_count == 2


@pytest.mark.parametrize("additional_variables", [[], ["downward_shortwave"]])
@pytest.mark.parametrize("derived_output_variables", [[], ["downwelling_shortwave"]])
def test_main_dumps_correct_predictor(
    tmpdir,
    mock_load_batches: mock.MagicMock,
    mock_train_dense_model: mock.MagicMock,
    additional_variables: Sequence[str],
    derived_output_variables: Sequence[str],
):
    """
    Test of fv3fit.train main function only, using mocks for training function
    and data loading.
    """
    artifacts = call_main(
        tmpdir, mock_load_batches, additional_variables, derived_output_variables,
    )
    mock_predictor = mock_train_dense_model.return_value
    if len(derived_output_variables) > 0:
        dump_predictor = artifacts.MockDerivedModel.return_value
    else:
        dump_predictor = mock_predictor
    dump_predictor.dump.assert_called_once()
    assert artifacts.output_path in dump_predictor.dump.call_args[0]


@pytest.mark.parametrize("additional_variables", [[], ["downward_shortwave"]])
@pytest.mark.parametrize("derived_output_variables", [[], ["downwelling_shortwave"]])
def test_main_uses_derived_model_only_if_needed(
    tmpdir,
    mock_load_batches: mock.MagicMock,
    mock_train_dense_model: mock.MagicMock,
    additional_variables: Sequence[str],
    derived_output_variables: Sequence[str],
):
    """
    Test of fv3fit.train main function only, using mocks for training function
    and data loading.
    """
    artifacts = call_main(
        tmpdir, mock_load_batches, additional_variables, derived_output_variables,
    )
    if len(derived_output_variables) > 0:
        artifacts.MockDerivedModel.assert_called_once_with(
            mock_train_dense_model.return_value, derived_output_variables
        )
    else:
        artifacts.MockDerivedModel.assert_not_called()


@pytest.mark.parametrize("additional_variables", [[], ["downward_shortwave"]])
@pytest.mark.parametrize("derived_output_variables", [[], ["downwelling_shortwave"]])
def test_main_calls_train_with_correct_arguments(
    tmpdir,
    mock_load_batches: mock.MagicMock,
    mock_train_dense_model: mock.MagicMock,
    additional_variables: Sequence[str],
    derived_output_variables: Sequence[str],
):
    """
    Test of fv3fit.train main function only, using mocks for training function
    and data loading.
    """
    artifacts = call_main(
        tmpdir, mock_load_batches, additional_variables, derived_output_variables,
    )
    mock_train_dense_model.assert_called_once_with(
        input_variables=artifacts.input_variables,
        output_variables=artifacts.output_variables,
        hyperparameters=artifacts.hyperparameters,
        train_batches=mock_load_batches.return_value,
        validation_batches=mock_load_batches.return_value,
    )


def get_config(
    tmpdir, additional_variables, derived_output_variables, model_type, hyperparameters,
):
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
    mock_dataset = get_mock_dataset(n_time=9)
    train_times = [vcm.encode_time(dt) for dt in mock_dataset["time"][:6].values]
    validation_times = [vcm.encode_time(dt) for dt in mock_dataset["time"][6:9].values]
    # TODO: refactor to use a loaders function that generates dummy data
    # instead of reading from disk, for CLI tests where we can't mock
    data_dir = os.path.join(base_dir, "data")
    mock_dataset.to_zarr(data_dir, consolidated=True)
    # assert ".zmetadata" in os.listdir(data_dir)
    train_data_config = loaders.BatchesConfig(
        function="batches_from_geodata",
        kwargs=dict(
            data_path=data_dir,
            mapping_function="open_zarr",
            timesteps=train_times,
            needs_grid=False,
            res="c8_random_values",
            timesteps_per_batch=3,
        ),
    )
    validation_data_config = loaders.BatchesConfig(
        function="batches_from_geodata",
        kwargs=dict(
            data_path=data_dir,
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
        training_config=training_filename,
        training_data_config=train_data_filename,
        validation_data_config=validation_data_filename,
        output_path=output_path,
    )
    return TestConfig(
        args, input_variables, output_variables, output_path, mock_dataset
    )


def cli_main(args: MainArgs):
    if args.validation_data_config is None:
        validation_args = []
    else:
        validation_args = ["--validation-data-config", args.validation_data_config]
    if args.local_download_path is None:
        local_download_args = []
    else:
        local_download_args = ["--local-download-path", args.local_download_path]
    subprocess.check_call(
        [
            "python",
            "-m",
            "fv3fit.train",
            args.training_config,
            args.training_data_config,
            args.output_path,
        ]
        + validation_args
        + local_download_args
    )


@pytest.mark.parametrize(
    "model_type, hyperparameters",
    [
        pytest.param(
            "sklearn_random_forest",
            {"max_depth": 4, "n_estimators": 2},
            id="random_forest",
        ),
        pytest.param(
            "DenseModel",
            {
                "width": 4,
                "depth": 3,
                "save_model_checkpoints": False,
                "fit_kwargs": {"batch_size": 100, "validation_samples": 384},
            },
            id="dense",
        ),
        pytest.param(
            "DenseModel",
            {
                "width": 4,
                "depth": 3,
                "save_model_checkpoints": True,
                "fit_kwargs": {"batch_size": 100, "validation_samples": 384},
            },
            id="dense_with_checkpoints",
        ),
    ],
)
@pytest.mark.parametrize("use_local_download_path", [True, False])
def test_cli(
    tmpdir, use_local_download_path: bool, model_type: str, hyperparameters,
):
    """
    Test of fv3fit.train main function only, using mocks for training function
    and data loading.
    """
    additional_variables = []
    config = get_config(tmpdir, additional_variables, [], model_type, hyperparameters)
    mock_load_batches.return_value = [config.mock_dataset for _ in range(6)]
    if use_local_download_path:
        config.args.local_download_path = os.path.join(str(tmpdir), "local_download")
    cli_main(config.args)
    fv3fit.load(config.args.output_path)
    if use_local_download_path:
        assert len(os.listdir(config.args.local_download_path)) > 0
