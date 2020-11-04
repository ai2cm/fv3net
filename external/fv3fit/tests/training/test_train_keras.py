from typing import Iterable, Sequence
import xarray as xr
import pytest
import logging
import loaders
import fv3fit
import numpy as np
import tempfile
import subprocess
import os

from fv3fit.keras.__main__ import _set_random_seed


logger = logging.getLogger(__name__)


@pytest.fixture(params=["DenseModel"])
def model_type(request) -> str:
    return request.param


@pytest.fixture(params=["mse"])
def loss(request) -> str:
    return request.param


@pytest.fixture
def hyperparameters(model_type, loss) -> dict:
    if model_type == "DenseModel":
        hyperparameters = {"width": 4, "depth": 3}
        if loss:
            hyperparameters["loss"] = loss
        return hyperparameters
    else:
        raise NotImplementedError(model_type)


@pytest.fixture
def model(
    model_type: str,
    input_variables: Iterable[str],
    output_variables: Iterable[str],
    hyperparameters: dict,
) -> fv3fit.keras.Model:
    return fv3fit.keras.get_model(
        model_type,
        loaders.SAMPLE_DIM_NAME,
        input_variables,
        output_variables,
        **hyperparameters,
    )


def test_reproducibility(
    input_variables,
    hyperparameters,
    training_batches: Sequence[xr.Dataset],
    output_variables: Iterable[str],
):
    batch_dataset_test = training_batches[0]

    _set_random_seed(0)
    model_0 = fv3fit.keras.get_model(
        "DenseModel",
        loaders.SAMPLE_DIM_NAME,
        input_variables,
        output_variables,
        **hyperparameters,
    )
    model_0.fit(training_batches)
    result_0 = model_0.predict(batch_dataset_test)

    _set_random_seed(0)
    model_1 = fv3fit.keras.get_model(
        "DenseModel",
        loaders.SAMPLE_DIM_NAME,
        input_variables,
        output_variables,
        **hyperparameters,
    )
    model_1.fit(training_batches)
    result_1 = model_1.predict(batch_dataset_test)

    xr.testing.assert_allclose(result_0, result_1, rtol=1e-03)


def test_training(
    model: fv3fit.keras.Model,
    training_batches: Sequence[xr.Dataset],
    output_variables: Iterable[str],
):
    model.fit(training_batches)
    batch_dataset = training_batches[0]
    result = model.predict(batch_dataset)
    validate_dataset_result(result, batch_dataset, output_variables)


def test_dump_and_load_before_training(
    model: fv3fit.keras.Model,
    training_batches: Sequence[xr.Dataset],
    output_variables: Iterable[str],
):
    with tempfile.TemporaryDirectory() as tmpdir:
        model.dump(tmpdir)
        model = model.__class__.load(tmpdir)
    model.fit(training_batches)
    batch_dataset = training_batches[0]
    result = model.predict(batch_dataset)
    validate_dataset_result(result, batch_dataset, output_variables)


def validate_dataset_result(
    result: xr.Dataset, batch_dataset: xr.Dataset, output_variables: Iterable[str]
):
    """
    Use assertions to test whether the predicted output dataset metadata matches
    metadata from a reference, for the given variable names. Also checks output values
    are present.
    """
    missing_names = set(output_variables).difference(result.data_vars.keys())
    assert len(missing_names) == 0
    for varname in output_variables:
        assert result[varname].shape == batch_dataset[varname].shape, varname
        assert np.sum(np.isnan(result[varname].values)) == 0


def test_dump_and_load_maintains_prediction(
    model: fv3fit.keras.Model,
    training_batches: Sequence[xr.Dataset],
    output_variables: Iterable[str],
):
    model.fit(training_batches)
    with tempfile.TemporaryDirectory() as tmpdir:
        model.dump(tmpdir)
        loaded_model = model.__class__.load(tmpdir)
    batch_dataset = training_batches[0]
    loaded_result = loaded_model.predict(batch_dataset)
    validate_dataset_result(loaded_result, batch_dataset, output_variables)
    original_result = model.predict(batch_dataset)
    xr.testing.assert_equal(loaded_result, original_result)


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
            "fv3fit.keras",
            data_source_path,
            train_config_filename,
            tmp_path,
        ]
    )
    required_names = ["model_data", "training_config.yml"]
    missing_names = set(required_names).difference(os.listdir(tmp_path))
    assert len(missing_names) == 0


@pytest.mark.parametrize(
    "loss, expected_loss",
    (
        pytest.param("mae", "mae", id="specified_loss"),
        pytest.param(None, "mse", id="default_loss"),
    ),
    indirect=["loss"],
)
def test_dump_and_load_loss_info_use_fixture(loss, expected_loss, model):
    with tempfile.TemporaryDirectory() as tmpdir:
        model.dump(tmpdir)
        model_loaded = model.__class__.load(tmpdir)
    assert model_loaded._loss == expected_loss
