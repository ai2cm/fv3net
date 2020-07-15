from typing import Iterable, Sequence
import xarray as xr
import pytest
import logging
from fv3fit._shared import ModelTrainingConfig
import numpy as np
import subprocess
import os
from fv3fit.sklearn._train import train_model


logger = logging.getLogger(__name__)


@pytest.fixture(params=["sklearn_random_forest"])
def model_type(request) -> str:
    return request.param


@pytest.fixture
def hyperparameters(model_type) -> dict:
    if model_type == "sklearn_random_forest":
        return {"max_depth": 4, "n_estimators": 2}
    else:
        raise NotImplementedError(model_type)


@pytest.mark.regression
def test_training(
    training_batches: Sequence[xr.Dataset],
    output_variables: Iterable[str],
    train_config: ModelTrainingConfig,
):
    model = train_model(training_batches, train_config)
    assert model.model.n_estimators == 2
    batch_dataset = training_batches[0]
    result = model.predict(batch_dataset, "sample")
    missing_names = set(output_variables).difference(result.data_vars.keys())
    assert len(missing_names) == 0
    for varname in output_variables:
        assert result[varname].shape == batch_dataset[varname].shape, varname
        assert np.sum(np.isnan(result[varname].values)) == 0


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
    subprocess.check_call(
        [
            "python",
            "-m",
            "fv3fit.sklearn",
            data_source_path,
            train_config_filename,
            tmp_path,
            "--no-train-subdir-append",
        ]
    )
    required_names = [
        "ML_model_training_report.html",
        "count_of_training_times_used.png",
        "sklearn_model.pkl",
        "training_config.yml",
    ]
    existing_names = os.listdir(tmp_path)
    missing_names = set(required_names).difference(existing_names)
    assert len(missing_names) == 0, existing_names
