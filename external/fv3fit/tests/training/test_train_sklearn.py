from typing import Iterable, Sequence
import xarray as xr
import pytest
import logging
from fv3fit._shared import ModelTrainingConfig
import numpy as np
import subprocess
import os
from vcm import safe

from fv3fit.sklearn._train import (
    train_model,
    _get_transformed_target_regressor,
    _get_target_scaler,
    _get_transformed_batch_regressor,
)
from fv3fit._shared import ArrayPacker, StandardScaler, ManualScaler
from sklearn.dummy import DummyRegressor

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
            "--no-train-subdir-append",
        ]
    )
    required_names = [
        "sklearn_model.pkl",
        "training_config.yml",
    ]
    existing_names = os.listdir(tmp_path)
    missing_names = set(required_names).difference(existing_names)
    assert len(missing_names) == 0, existing_names


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
            np.array([[1.0, 1.0], [2.0, 2.0]]),
        ),
    }
)


@pytest.mark.parametrize(
    "scaler_type, scaler_kwargs, delp, expected_y_normalized,",
    (
        ["standard", None, None, np.array([[-1.0, -1.0, 1.0], [1.0, 1.0, -1.0]])],
        [
            "mass",
            {"variable_scale_factors": {"y0": 200}},
            np.array([2.0, 2.0]),
            np.array([[10.0, 10.0, -1.0], [20.0, 20, -2.0]]),
        ],
    ),
)
def test__get_transformed_target_regressor(
    scaler_type, scaler_kwargs, delp, expected_y_normalized
):
    sample_dim = "sample"
    output_vars = ["y0", "y1"]
    regressor = DummyRegressor(strategy="mean")
    if delp is not None:
        norm_data["pressure_thickness_of_atmospheric_layer"] = (
            ["sample", "z"],
            np.array([delp, delp]),
        )
    transformed_target_regressor = _get_transformed_target_regressor(
        regressor,
        output_vars,
        norm_data,
        scaler_type=scaler_type,
        scaler_kwargs=scaler_kwargs,
    )
    packer = ArrayPacker(sample_dim, output_vars)
    y = packer.to_array(safe.get_variables(norm_data, output_vars))
    # normalize
    np.testing.assert_array_almost_equal(
        transformed_target_regressor.func(y), expected_y_normalized
    )
    # denormalize
    np.testing.assert_array_almost_equal(
        transformed_target_regressor.inverse_func(expected_y_normalized), y
    )


def test_same_scaler_after_fitting(training_batches, train_config):
    batch_0 = training_batches[0]
    packer = ArrayPacker("sample", train_config.output_variables)
    y = packer.to_array(safe.get_variables(batch_0, train_config.output_variables))
    model_wrapper = _get_transformed_batch_regressor(train_config, batch_0)
    normed_y = model_wrapper.model.base_regressor.func(y)
    model_wrapper.fit(batch_0)
    np.testing.assert_array_almost_equal(
        model_wrapper.model.base_regressor.func(y), normed_y
    )
