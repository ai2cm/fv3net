from typing import Iterable, Sequence
import xarray as xr
import pytest
import logging
from fv3fit._shared import ModelTrainingConfig
import numpy as np
import subprocess

from fv3fit.sklearn._train import (
    train_model,
    _get_target_scaler,
)
import fv3fit.sklearn
from fv3fit._shared import StandardScaler, ManualScaler

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


def test_training(
    training_batches: Sequence[xr.Dataset],
    output_variables: Iterable[str],
    train_config: ModelTrainingConfig,
):
    model = train_model(training_batches, train_config)
    assert model.model.n_estimators == 2
    batch_dataset = training_batches[0]
    result = model.predict(batch_dataset)
    missing_names = set(output_variables).difference(result.data_vars.keys())
    assert len(missing_names) == 0
    for varname in output_variables:
        assert result[varname].shape == batch_dataset[varname].shape, varname
        assert np.sum(np.isnan(result[varname].values)) == 0


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
        ]
    )

    fv3fit.sklearn.SklearnWrapper.load(str(tmp_path / "sklearn.yaml"))


@pytest.mark.parametrize(
    "scaler_type, expected_type", (["standard", StandardScaler], ["mass", ManualScaler])
)
def test__get_target_scaler_type(scaler_type, expected_type):
    scaler = _get_target_scaler(
        scaler_type, scaler_kwargs={}, norm_data=norm_data, output_vars=["y0", "y1"]
    )
    assert isinstance(scaler, expected_type)


norm_data = xr.Dataset(
    {
        "y0": (["sample", "z"], np.array([[1.0, 1.0], [2.0, 2.0]])),
        "y1": (["sample"], np.array([-1.0, -2.0])),
        "pressure_thickness_of_atmospheric_layer": (
            ["sample", "z"],
            np.array([[1.0, 1.0], [1.0, 1.0]]),
        ),
    }
)
