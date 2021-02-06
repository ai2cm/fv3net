from typing import Iterable, Sequence
import xarray as xr
import pytest
import logging
from fv3fit._shared import ModelTrainingConfig
import numpy as np
import subprocess
import copy


from fv3fit.keras import DummyModel

from synth import data_source_path


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




def test_offline_diags_integration(
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
            "fv3fit.train",
            data_source_path,
            train_config_filename,
            tmp_path,
        ]
    )

    fv3fit.sklearn.SklearnWrapper.load(str(tmp_path))
