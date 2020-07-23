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


logger = logging.getLogger(__name__)


@pytest.fixture(params=["DenseModel"])
def model_type(request) -> str:
    return request.param


@pytest.fixture
def hyperparameters(model_type) -> dict:
    if model_type == "DenseModel":
        return {"width": 8, "depth": 3}
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


@pytest.mark.regression
def test_training(
    model: fv3fit.keras.Model,
    training_batches: Sequence[xr.Dataset],
    output_variables: Iterable[str],
):
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


@pytest.mark.regression
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
            "fv3fit.keras",
            data_source_path,
            train_config_filename,
            tmp_path,
            "--no-train-subdir-append",
        ]
    )
    required_names = ["model_data", "training_config.yml"]
    missing_names = set(required_names).difference(os.listdir(tmp_path))
    assert len(missing_names) == 0
