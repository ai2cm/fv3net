from typing import Callable, Optional, TextIO
from fv3fit.typing import Dataclass
import pytest
import fv3fit.train
import xarray as xr
import numpy as np
from fv3fit._shared.config import ESTIMATORS, get_hyperparameter_class
import vcm.testing
import tempfile


# automatically test on every registered training class
@pytest.fixture(params=list(ESTIMATORS.keys()))
def model_type(request):
    return request.param


def test_training_functions_exist():
    assert len(ESTIMATORS.keys()) > 0


def assert_can_learn_identity(
    model_type,
    hyperparameters: Dataclass,
    sample_func: Callable[[], xr.DataArray],
    max_rmse: float,
    regtest: Optional[TextIO] = None,
):
    """
    Args:
        model_type: type of model to train
        hyperparameters: model configuration
        sample_func: function that returns example DataArrays for training and
            validation, should return different data on subsequent calls
        max_rmse: maximum permissible root mean squared error
        regtest: if given, write hash of output dataset to this file object
    """
    hyperparameters = get_hyperparameter_class(model_type)()
    input_variables = ["var_in"]
    output_variables = ["var_out"]
    data_array = sample_func()
    train_dataset = xr.Dataset(data_vars={"var_in": data_array, "var_out": data_array})
    train_batches = [train_dataset for _ in range(10)]
    val_batches = []
    model = fv3fit.train.fit_model(
        model_type,
        input_variables,
        output_variables,
        hyperparameters,
        train_batches,
        val_batches,
    )
    data_array = sample_func()
    test_dataset = xr.Dataset(data_vars={"var_in": data_array, "var_out": data_array})
    out_dataset = model.predict(test_dataset)
    rmse = np.mean((out_dataset["var_out"] - test_dataset["var_out"]) ** 2) ** 0.5
    assert rmse < max_rmse
    if regtest is not None:
        for result in vcm.testing.checksum_dataarray_mapping(test_dataset):
            print(result, file=regtest)
        for result in vcm.testing.checksum_dataarray_mapping(out_dataset):
            print(result, file=regtest)


def test_train_default_model_on_identity(model_type, regtest):
    """
    The model with default configuration options can learn the identity function,
    using gaussian-sampled data around 0 with unit variance.
    """
    fv3fit.train.set_random_seed(1)
    random = np.random.RandomState(0)
    # don't set n_feature too high for this, because of curse of dimensionality
    n_sample, n_feature = int(5e3), 2
    hyperparameters = get_hyperparameter_class(model_type)()

    def sample_func():
        return xr.DataArray(
            random.randn(n_sample, n_feature), dims=["sample", "feature_dim"]
        )

    assert_can_learn_identity(
        model_type,
        hyperparameters=hyperparameters,
        sample_func=sample_func,
        max_rmse=0.05,
        regtest=regtest,
    )


def test_train_default_model_on_nonstandard_identity(model_type):
    """
    The model with default configuration options can learn the identity function,
    using gaussian-sampled data around a non-zero value with non-unit variance.
    """
    mean = 100.0
    std = 15.0
    random = np.random.RandomState(0)
    # don't set n_feature too high for this, because of curse of dimensionality
    n_sample, n_feature = int(5e3), 2
    hyperparameters = get_hyperparameter_class(model_type)()

    def sample_func():
        return xr.DataArray(
            random.randn(n_sample, n_feature) * std + mean,
            dims=["sample", "feature_dim"],
        )

    assert_can_learn_identity(
        model_type,
        hyperparameters=hyperparameters,
        sample_func=sample_func,
        max_rmse=0.05 * std,
    )


def test_dump_and_load_default_maintains_prediction(model_type):
    hyperparameters = get_hyperparameter_class(model_type)()
    input_variables = ["var_in"]
    output_variables = ["var_out"]
    random = np.random.RandomState(0)
    n_sample, n_feature = 500, 2

    def sample_func():
        return xr.DataArray(
            random.randn(n_sample, n_feature), dims=["sample", "feature_dim"]
        )

    data_array = sample_func()
    train_dataset = xr.Dataset(data_vars={"var_in": data_array, "var_out": data_array})
    train_batches = [train_dataset for _ in range(10)]
    val_batches = []
    model = fv3fit.train.fit_model(
        model_type,
        input_variables,
        output_variables,
        hyperparameters,
        train_batches,
        val_batches,
    )
    data_array = sample_func()
    test_dataset = xr.Dataset(data_vars={"var_in": data_array, "var_out": data_array})
    original_result = model.predict(test_dataset)
    with tempfile.TemporaryDirectory() as tmpdir:
        model.dump(tmpdir)
        loaded_model = model.__class__.load(tmpdir)
    loaded_result = loaded_model.predict(test_dataset)
    xr.testing.assert_equal(loaded_result, original_result)
