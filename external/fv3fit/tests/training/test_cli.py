from fv3fit._shared import _ModelTrainingConfig as ModelTrainingConfig
import fv3fit
import yaml
import pytest
import tempfile
import subprocess
import os


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
    "validation_timesteps", [True, False],
)
def test_local_download_path(
    model_info, data_info, validation_timesteps, tmp_path: str,
):
    """
    Test the bash endpoint for training the model produces the expected output files.
    """
    config = ModelTrainingConfig(
        data_path=data_info["data_path"],
        model_type=model_info["model_type"],
        hyperparameters=model_info["hyperparameters"],
        input_variables=["air_temperature", "specific_humidity"],
        output_variables=["dQ1", "dQ2"],
        batch_function="batches_from_geodata",
        batch_kwargs=data_info["batch_kwargs"],
        scaler_type="standard",
        scaler_kwargs={},
        additional_variables=[],
        random_seed=0,
        validation_timesteps=data_info["validation_timesteps"]
        if validation_timesteps
        else [],
    )

    with tempfile.NamedTemporaryFile(mode="w") as f:
        yaml.dump(config.asdict(), f)
        with tempfile.TemporaryDirectory() as tmpdir:
            subprocess.check_call(
                [
                    "python",
                    "-m",
                    "fv3fit.train",
                    config.data_path,
                    f.name,
                    tmp_path,
                    "--local-download-path",
                    tmpdir,
                ]
            )
            assert len(os.listdir(tmpdir)) > 0
        fv3fit.load(str(tmp_path))
        fv3fit._shared.config.load_training_config(str(tmp_path))


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
    "validation_timesteps", [True, False],
)
def test_training_integration(
    model_info, data_info, validation_timesteps, tmp_path: str,
):
    """
    Test the bash endpoint for training the model produces the expected output files.
    """
    config = ModelTrainingConfig(
        data_path=data_info["data_path"],
        model_type=model_info["model_type"],
        hyperparameters=model_info["hyperparameters"],
        input_variables=["air_temperature", "specific_humidity"],
        output_variables=["dQ1", "dQ2"],
        batch_function="batches_from_geodata",
        batch_kwargs=data_info["batch_kwargs"],
        scaler_type="standard",
        scaler_kwargs={},
        additional_variables=[],
        random_seed=0,
        validation_timesteps=data_info["validation_timesteps"]
        if validation_timesteps
        else [],
    )

    with tempfile.NamedTemporaryFile(mode="w") as f:
        yaml.dump(config.asdict(), f)
        subprocess.check_call(
            ["python", "-m", "fv3fit.train", config.data_path, f.name, tmp_path]
        )
        fv3fit.load(str(tmp_path))
        fv3fit._shared.config.load_training_config(str(tmp_path))
