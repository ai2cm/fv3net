import dataclasses
import fv3fit
import loaders
import yaml
import pytest
import subprocess
import os
import cftime
import xarray
import numpy as np


@pytest.fixture
def data_info(tmpdir):
    data_dir = os.path.join(str(tmpdir), "data")
    x, y, z, tile, time = (8, 8, 79, 6, 2)
    arr = np.zeros((time, tile, z, y, x))
    arr_surf = np.zeros((time, tile, y, x))
    dims = ["time", "tile", "z", "y", "x"]
    dims_surf = ["time", "tile", "y", "x"]

    data = xarray.Dataset(
        {
            "specific_humidity": (dims, arr),
            "air_temperature": (dims, arr),
            "downward_shortwave": (dims_surf, arr_surf),
            "net_shortwave": (dims_surf, arr_surf),
            "downward_longwave": (dims_surf, arr_surf),
            "dQ1": (dims, arr),
            "dQ2": (dims, arr),
            "dQu": (dims, arr),
            "dQv": (dims, arr),
        },
        coords={
            "time": [
                cftime.DatetimeJulian(2016, 8, 1),
                cftime.DatetimeJulian(2016, 8, 2),
            ]
        },
    )

    data.to_zarr(data_dir, consolidated=True)
    return dict(
        data_path=data_dir,
        batches_kwargs=dict(
            mapping_function="open_zarr",
            needs_grid=False,
            res="c8_random_values",
            timesteps_per_batch=1,
        ),
        train_timesteps=["20160801.000000"],
        validation_timesteps=["20160802.000000"],
    )


@pytest.mark.parametrize(
    "model_info",
    [
        dict(
            model_type="sklearn_random_forest",
            hyperparameters={"max_depth": 4, "n_estimators": 2},
        ),
        dict(
            model_type="DenseModel",
            hyperparameters={
                "width": 4,
                "depth": 3,
                "fit_kwargs": {"batch_size": 100, "validation_samples": 384},
            },
        ),
        dict(
            model_type="DenseModel",
            hyperparameters={
                "width": 4,
                "depth": 3,
                "fit_kwargs": {"batch_size": 100, "validation_samples": 384},
            },
            save_model_checkpoints=True,
        ),
    ],
)
@pytest.mark.parametrize(
    "use_validation_data", [True, False],
)
def test_training_integration(
    model_info, data_info, use_validation_data: bool, tmp_path: str,
):
    """
    Test the bash endpoint for training the model produces the expected output files.
    """
    (
        training_filename,
        train_data_filename,
        validation_data_filename,
        output_path,
    ) = get_config(model_info, data_info, tmp_path, use_validation_data)

    if use_validation_data:
        validation_args = [
            "--validation-data-config",
            validation_data_filename,
        ]
    else:
        validation_args = []

    subprocess.check_call(
        [
            "python",
            "-m",
            "fv3fit.train",
            training_filename,
            train_data_filename,
            output_path,
        ]
        + validation_args
    )
    fv3fit.load(output_path)


def get_config(model_info, data_info, tmp_path, use_validation_data: bool):
    """
    Initialize configuration files and get paths required to run fv3fit.train
    """
    training_config = fv3fit.TrainingConfig(
        model_type=model_info["model_type"],
        input_variables=["air_temperature", "specific_humidity"],
        output_variables=["dQ1", "dQ2"],
        hyperparameters=fv3fit.get_hyperparameter_class(model_info["model_type"])(
            **model_info["hyperparameters"]
        ),
        additional_variables=[],
    )
    train_data_config = loaders.BatchesConfig(
        function="batches_from_geodata",
        kwargs=dict(
            data_path=data_info["data_path"],
            timesteps=data_info["train_timesteps"],
            **data_info["batches_kwargs"]
        ),
    )
    if use_validation_data:
        validation_timesteps = data_info["validation_timesteps"]
    else:
        validation_timesteps = []
    validation_data_config = loaders.BatchesConfig(
        function="batches_from_geodata",
        kwargs=dict(
            data_path=data_info["data_path"],
            timesteps=validation_timesteps,
            **data_info["batches_kwargs"]
        ),
    )

    train_data_filename = os.path.join(tmp_path, "train_data.yaml")
    validation_data_filename = os.path.join(tmp_path, "validation_data.yaml")
    training_filename = os.path.join(tmp_path, "training.yaml")
    with open(train_data_filename, "w") as f:
        yaml.dump(dataclasses.asdict(train_data_config), f)
    with open(validation_data_filename, "w") as f:
        yaml.dump(dataclasses.asdict(validation_data_config), f)
    with open(training_filename, "w") as f:
        yaml.dump(dataclasses.asdict(training_config), f)
    output_path = os.path.join(tmp_path, "output")
    return training_filename, train_data_filename, validation_data_filename, output_path


@pytest.mark.parametrize(
    "model_info",
    [
        dict(
            model_type="sklearn_random_forest",
            hyperparameters={"max_depth": 4, "n_estimators": 2},
        ),
    ],
)
@pytest.mark.parametrize(
    "use_validation_data", [True, False],
)
def test_local_download_path(
    model_info, data_info, use_validation_data: bool, tmp_path,
):
    """
    Test the bash endpoint for training the model produces the expected output files.
    """
    (
        training_filename,
        train_data_filename,
        validation_data_filename,
        output_path,
    ) = get_config(model_info, data_info, tmp_path, use_validation_data)
    local_download_path = os.path.join(tmp_path, "local_data")

    if use_validation_data:
        validation_args = [
            "--validation-data-config",
            validation_data_filename,
        ]
    else:
        validation_args = []

    subprocess.check_call(
        [
            "python",
            "-m",
            "fv3fit.train",
            training_filename,
            train_data_filename,
            output_path,
            "--local-download-path",
            local_download_path,
        ]
        + validation_args
    )
    assert len(os.listdir(local_download_path)) > 0
    fv3fit.load(output_path)
