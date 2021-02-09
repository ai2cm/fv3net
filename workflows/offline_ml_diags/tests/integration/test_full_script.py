from typing import Iterable, Sequence
import xarray as xr
import pytest
import logging
from fv3fit._shared import ModelTrainingConfig
import numpy as np
import subprocess
import copy


import fv3fit
from synth import data_source_path


logger = logging.getLogger(__name__)



@pytest.fixture
def model(
    model_type: str,
    input_variables: Iterable[str],
    output_variables: Iterable[str],
    hyperparameters: dict,
) -> fv3fit.Estimator:
    fit_kwargs = hyperparameters.pop("fit_kwargs", {})
    return fv3fit.keras.get_model(
        "DenseModel",
        loaders.SAMPLE_DIM_NAME,
        input_variables,
        output_variables,
        width=3,
        depth=2
    )
         

def test_offline_diags_integration(
    data_source_path: str,
    train_config_filename: str,
    data_source_name: str,
):
    """
    Test the bash endpoint for computing offline diagnostics
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = os.path.join(tmpdir, "trained_model")
        model.dump(model_dir)
        train_config.dump(os.path.join(model_dir, "training_config.yml")
        subprocess.check_call(
            [
                "python",
                "-m",
                "offline_ml_diags.compute_diags",
                model_dir,
                os.path.join(tmpdir, "offline_diags"),
                "--timesteps-n-samples 2"
            ]
        )
