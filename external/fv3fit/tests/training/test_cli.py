from fv3fit._shared import ModelTrainingConfig
import fv3fit
from conftest import get_batch_kwargs
import yaml
import pytest
import tempfile
import subprocess
import os


@pytest.mark.parametrize(
    "validation_timesteps", [["20160801.003000"], None,],
)
def test_training_integration(
    data_source_path, data_source_name, validation_timesteps, tmp_path: str,
):
    """
    Test the bash endpoint for training the model produces the expected output files.
    """

    config = ModelTrainingConfig(
        data_path="train_data_path",
        model_type="DenseModel",
        hyperparameters={
            "width": 4,
            "depth": 3,
            "fit_kwargs": {"batch_size": 100, "validation_samples": 384},
        },
        input_variables=["air_temperature", "specific_humidity"],
        output_variables=["dQ1", "dQ2"],
        batch_function="batches_from_geodata",
        batch_kwargs=get_batch_kwargs(data_source_name),
        scaler_type="standard",
        scaler_kwargs={},
        additional_variables=None,
        random_seed=0,
        validation_timesteps=validation_timesteps,
    )

    with tempfile.NamedTemporaryFile(mode="w") as f:
        yaml.dump(config.asdict(), f)

        subprocess.check_call(
            ["python", "-m", "fv3fit.train", data_source_path, f.name, tmp_path,]
        )
        fv3fit.load(str(tmp_path))
        fv3fit.load_training_config(str(tmp_path))
