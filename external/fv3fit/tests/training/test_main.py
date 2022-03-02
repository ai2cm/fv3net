import dataclasses
import subprocess
from typing import Any, Optional, Sequence
import fv3fit
from fv3fit._shared.config import get_hyperparameter_class
from fv3fit._shared.hyperparameters import Hyperparameters
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


def get_mock_dataset(n_time, unstacked_dims: Sequence[str]):
    shape = tuple(range(4, 4 + len(unstacked_dims) + 1))
    dims = tuple(["sample"] + list(unstacked_dims))
    arr = np.zeros(shape)
    shape_surface = shape[:-1]
    dims_surface = dims[:-1]
    arr_surface = np.zeros(shape_surface)

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
    variables: Sequence[str]
    hyperparameters: Hyperparameters
    output_path: str
    mock_dataset: xr.Dataset


@dataclasses.dataclass
class CallArtifacts:
    output_path: str
    variables: Sequence[str]
    MockDerivedModel: mock.Mock
    hyperparameters: Any


@pytest.fixture
def mock_train_dense_model():
    original_func = fv3fit.get_training_function("dense")
    train_mock = mock.MagicMock(name="train_dense_model", spec=original_func)
    train_mock.return_value = mock.MagicMock(
        name="train_dense_model_return", spec=fv3fit.Predictor
    )
    register("mock")(train_mock.return_value.__class__)
    try:
        fv3fit._shared.config.register_training_function(
            "dense", fv3fit.DenseHyperparameters
        )(train_mock)
        yield train_mock
    finally:
        fv3fit._shared.config.register_training_function(
            "dense", fv3fit.DenseHyperparameters
        )(original_func)
        register._model_types.pop("mock")


@pytest.fixture
def mock_load_batches():
    magic_load_mock = mock.MagicMock(name="load_batches")
    with mock.patch.object(loaders.BatchesConfig, "load_batches", magic_load_mock):
        yield magic_load_mock


@pytest.fixture
def mock_batches_from_netcdf():
    with mock.patch(
        "loaders.batches_from_netcdf", spec=loaders.batches_from_netcdf
    ) as magic_load_mock:
        yield magic_load_mock


def call_main(
    tmpdir,
    mock_load_batches,
    derived_output_variables,
    use_validation_data: bool,
    use_local_download_path: bool = False,
):
    model_type = "dense"
    hyperparameters_dict = {}
    config = get_config(
        tmpdir,
        derived_output_variables,
        model_type,
        hyperparameters_dict,
        use_validation_data,
        use_local_download_path=use_local_download_path,
        unstacked_dims=["z"],
    )
    mock_load_batches.return_value = [config.mock_dataset for _ in range(6)]
    with mock.patch("fv3fit.DerivedModel") as MockDerivedModel:
        MockDerivedModel.return_value = mock.MagicMock(
            name="derived_model_return", spec=fv3fit.Predictor
        )
        fv3fit.train.main(config.args)
    return CallArtifacts(
        config.output_path, config.variables, MockDerivedModel, config.hyperparameters,
    )


@pytest.mark.parametrize("derived_output_variables", [[], ["downwelling_shortwave"]])
@pytest.mark.parametrize("use_validation_data", [True, False])
def test_main_calls_load_batches_correctly(
    tmpdir,
    mock_load_batches: mock.MagicMock,
    mock_train_dense_model: mock.MagicMock,
    derived_output_variables: Sequence[str],
    use_validation_data: bool,
):
    """
    Test of fv3fit.train main function only, using mocks for training function
    and data loading.
    """
    artifacts = call_main(
        tmpdir, mock_load_batches, derived_output_variables, use_validation_data,
    )
    mock_load_batches.assert_called_with(variables=artifacts.variables)
    if use_validation_data:
        assert (
            mock_load_batches.call_args_list[0] == mock_load_batches.call_args_list[1]
        )
        assert mock_load_batches.call_count == 2
    else:
        assert mock_load_batches.call_count == 1


@pytest.mark.parametrize("derived_output_variables", [[], ["downwelling_shortwave"]])
def test_main_dumps_correct_predictor(
    tmpdir,
    mock_load_batches: mock.MagicMock,
    mock_train_dense_model: mock.MagicMock,
    derived_output_variables: Sequence[str],
):
    artifacts = call_main(
        tmpdir, mock_load_batches, derived_output_variables, use_validation_data=True,
    )
    mock_predictor = mock_train_dense_model.return_value
    if len(derived_output_variables) > 0:
        dump_predictor = artifacts.MockDerivedModel.return_value
    else:
        dump_predictor = mock_predictor
    dump_predictor.dump.assert_called_once()
    assert artifacts.output_path in dump_predictor.dump.call_args[0]


@pytest.mark.parametrize("derived_output_variables", [[], ["downwelling_shortwave"]])
def test_main_uses_derived_model_only_if_needed(
    tmpdir,
    mock_load_batches: mock.MagicMock,
    mock_train_dense_model: mock.MagicMock,
    derived_output_variables: Sequence[str],
):
    artifacts = call_main(
        tmpdir, mock_load_batches, derived_output_variables, use_validation_data=True,
    )
    if len(derived_output_variables) > 0:
        artifacts.MockDerivedModel.assert_called_once_with(
            mock_train_dense_model.return_value, derived_output_variables
        )
    else:
        artifacts.MockDerivedModel.assert_not_called()


@pytest.mark.parametrize("derived_output_variables", [[], ["downwelling_shortwave"]])
@pytest.mark.parametrize("use_validation_data", [True, False])
@pytest.mark.parametrize("use_local_download_path", [True, False])
def test_main_calls_train_with_correct_arguments(
    tmpdir,
    mock_load_batches: mock.MagicMock,
    mock_batches_from_netcdf: mock.MagicMock,
    mock_train_dense_model: mock.MagicMock,
    derived_output_variables: Sequence[str],
    use_validation_data: bool,
    use_local_download_path: bool,
):
    artifacts = call_main(
        tmpdir,
        mock_load_batches,
        derived_output_variables,
        use_validation_data,
        use_local_download_path=use_local_download_path,
    )
    if use_local_download_path:
        mock_batches = mock_batches_from_netcdf.return_value
    else:
        mock_batches = mock_load_batches.return_value
    if use_validation_data:
        validation_batches = mock_batches
    else:
        validation_batches = []
    mock_train_dense_model.assert_called_once_with(
        hyperparameters=artifacts.hyperparameters,
        train_batches=mock_batches,
        validation_batches=validation_batches,
    )


def get_hyperparameters(
    model_type, hyperparameter_dict, input_variables, output_variables
):
    cls = get_hyperparameter_class(model_type)
    try:
        hyperparameters = cls(**hyperparameter_dict)
    except TypeError:
        hyperparameters = cls(
            input_variables=input_variables,
            output_variables=output_variables,
            **hyperparameter_dict
        )
    return hyperparameters


def get_config(
    tmpdir,
    derived_output_variables,
    model_type,
    hyperparameter_dict,
    use_validation_data: bool,
    unstacked_dims: Sequence[str],
    use_local_download_path: bool = False,
):
    base_dir = str(tmpdir)
    input_variables = ["air_temperature", "specific_humidity"]
    output_variables = ["dQ1", "dQ2"]
    all_variables = input_variables + output_variables
    hyperparameters = get_hyperparameters(
        model_type, hyperparameter_dict, input_variables, output_variables
    )
    training_config = fv3fit.TrainingConfig(
        model_type=model_type,
        hyperparameters=hyperparameters,
        derived_output_variables=derived_output_variables,
    )
    mock_dataset = get_mock_dataset(n_time=9, unstacked_dims=unstacked_dims)
    train_times = [vcm.encode_time(dt) for dt in mock_dataset["time"][:6].values]
    validation_times = [vcm.encode_time(dt) for dt in mock_dataset["time"][6:9].values]
    # TODO: refactor to use a loaders function that generates dummy data
    # instead of reading from disk, for CLI tests where we can't mock
    data_path = os.path.join(base_dir, "data")
    mock_dataset.to_zarr(data_path, consolidated=True)
    train_data_config = loaders.BatchesConfig(
        function="batches_from_geodata",
        kwargs=dict(
            data_path=data_path,
            variable_names=all_variables,
            mapping_function="open_zarr",
            timesteps=train_times,
            needs_grid=False,
            res="c8_random_values",
            timesteps_per_batch=3,
            unstacked_dims=unstacked_dims,
        ),
    )
    if use_validation_data:
        validation_data_config = loaders.BatchesConfig(
            function="batches_from_geodata",
            kwargs=dict(
                data_path=data_path,
                variable_names=all_variables,
                mapping_function="open_zarr",
                timesteps=validation_times,
                needs_grid=False,
                res="c8_random_values",
                timesteps_per_batch=3,
                unstacked_dims=unstacked_dims,
            ),
        )
        validation_data_filename = os.path.join(base_dir, "validation_data.yaml")
        with open(validation_data_filename, "w") as f:
            yaml.dump(dataclasses.asdict(validation_data_config), f)
    else:
        validation_data_filename = None
    train_data_filename = os.path.join(base_dir, "train_data.yaml")
    training_filename = os.path.join(base_dir, "training.yaml")
    with open(train_data_filename, "w") as f:
        yaml.dump(dataclasses.asdict(train_data_config), f)
    with open(training_filename, "w") as f:
        yaml.dump(dataclasses.asdict(training_config), f)
    output_path = os.path.join(base_dir, "output")

    if use_local_download_path:
        local_download_path = os.path.join(base_dir, "local_data")
    else:
        local_download_path = None

    args = MainArgs(
        data_path=data_path,
        training_config=training_filename,
        training_data_config=train_data_filename,
        validation_data_config=validation_data_filename,
        output_path=output_path,
        local_download_path=local_download_path,
    )
    return TestConfig(
        args, training_config.variables, hyperparameters, output_path, mock_dataset
    )


def test_train_config_override_args(tmpdir, mock_load_batches, mock_train_dense_model):
    model_type = "dense"
    hyperparameter_dict = {
        "dense_network": {"width": 6},
        "optimizer_config": {"name": "MyOpt", "kwargs": {"lr": 0.01}},
    }
    config = get_config(
        tmpdir,
        derived_output_variables=[],
        model_type=model_type,
        hyperparameter_dict=hyperparameter_dict,
        use_validation_data=True,
        unstacked_dims=["z"],
    )
    mock_load_batches.return_value = [config.mock_dataset for _ in range(6)]
    assert config.hyperparameters.dense_network["width"] == 6
    assert config.hyperparameters.optimizer_config["kwargs"]["lr"] == 0.01
    patch_args = [
        "--hyperparameters.dense_network.width",
        "12",
        "--hyperparameters.optimizer_config.kwargs.lr",
        "0.02",
    ]
    fv3fit.train.main(config.args, unknown_args=patch_args)
    mock_train_dense_model.assert_called_once()
    hyperparameters = mock_train_dense_model.call_args[1]["hyperparameters"]
    assert hyperparameters.dense_network.width == 12
    assert hyperparameters.optimizer_config.kwargs["lr"] == 0.02
    assert hyperparameters.optimizer_config.name == "MyOpt"


def cli_main(args: MainArgs):
    if args.validation_data_config is None:
        validation_args = []
    else:
        validation_args = ["--validation-data-config", args.validation_data_config]
    if args.local_download_path is None:
        local_download_args = []
    else:
        local_download_args = ["--cache.local_download_path", args.local_download_path]
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
    # if you need pdb support, temporarily replace the check_call above with this:
    # fv3fit.train.main(args)


@pytest.mark.parametrize(
    "model_type, hyperparameter_dict",
    [
        pytest.param(
            "sklearn_random_forest",
            {"max_depth": 4, "n_estimators": 2},
            id="random_forest",
        ),
        pytest.param("dense", {"save_model_checkpoints": False}, id="dense"),
        pytest.param(
            "dense", {"save_model_checkpoints": True}, id="dense_with_checkpoints",
        ),
    ],
)
@pytest.mark.parametrize(
    "use_local_download_path, use_validation_data",
    [(False, False), (False, True), (True, False)],
)
def test_cli(
    tmpdir,
    use_local_download_path: bool,
    use_validation_data: bool,
    model_type: str,
    hyperparameter_dict,
):
    """
    Test of fv3fit.train command-line interface.
    """
    config = get_config(
        tmpdir,
        [],
        model_type,
        hyperparameter_dict,
        use_validation_data,
        use_local_download_path=use_local_download_path,
        unstacked_dims=["z"],
    )
    if not use_local_download_path and model_type == "dense":
        # dense requires caching if unstacked_dims is set as double-stacking
        # a MultiIndex causes an error, can remove this block when dense training
        # no longer stacks
        with pytest.raises(
            expected_exception=(ValueError, subprocess.CalledProcessError)
        ):
            cli_main(config.args)
    else:
        cli_main(config.args)
        fv3fit.load(config.args.output_path)
        if use_local_download_path:
            assert len(os.listdir(config.args.local_download_path)) > 0
