from typing import Iterable, Sequence
import xarray as xr
import pytest
import logging
from fv3fit._shared import ModelTrainingConfig
import numpy as np
import subprocess
import copy


from fv3fit.sklearn._train import get_transformed_batch_regressor
import fv3fit.sklearn

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
    model = get_transformed_batch_regressor(train_config)
    model.fit(training_batches)
    # This is the number of random forests in the ensemble, not the
    # number of total trees across the ensemble
    assert model.model.n_estimators == 1

    # assert that the target scaler is fitted
    assert model.target_scaler is not None

    batch_dataset = training_batches[0]
    result = model.predict(batch_dataset)
    missing_names = set(output_variables).difference(result.data_vars.keys())
    assert len(missing_names) == 0
    for varname in output_variables:
        assert result[varname].shape == batch_dataset[varname].shape, varname
        assert np.sum(np.isnan(result[varname].values)) == 0


def test_reproducibility(
    training_batches: Sequence[xr.Dataset], train_config: ModelTrainingConfig,
):
    batch_dataset = training_batches[0]
    train_config.hyperparameters["random_state"] = 0

    model_0 = get_transformed_batch_regressor(train_config)
    model_0.fit(copy.deepcopy(training_batches))
    result_0 = model_0.predict(batch_dataset)

    model_1 = get_transformed_batch_regressor(train_config)
    model_1.fit(copy.deepcopy(training_batches))
    result_1 = model_1.predict(batch_dataset)

    xr.testing.assert_allclose(result_0, result_1)


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
            "fv3fit.train",
            "sklearn",
            data_source_path,
            train_config_filename,
            tmp_path,
        ]
    )

    fv3fit.sklearn.SklearnWrapper.load(str(tmp_path))
