from typing import Callable, Optional, TextIO
from fv3fit.typing import Dataclass
import pytest
import xarray as xr
import numpy as np
import fv3fit
from fv3fit._shared.config import TRAINING_FUNCTIONS, get_hyperparameter_class
import vcm.testing
import tempfile


# automatically test on every registered training class
@pytest.fixture(params=list(TRAINING_FUNCTIONS.keys()))
def model_type(request):
    return request.param


SYSTEM_DEPENDENT_TYPES = ["DenseModel", "sklearn_random_forest"]
"""model types which produce different results on different systems"""


def test_training_functions_exist():
    assert len(TRAINING_FUNCTIONS.keys()) > 0


def train_identity_model(model_type, sample_func, hyperparameters):
    input_variables = ["var_in"]
    output_variables = ["var_out"]
    data_array = sample_func()
    train_dataset = xr.Dataset(data_vars={"var_in": data_array, "var_out": data_array})
    train_batches = [train_dataset for _ in range(10)]
    val_batches = []
    train = fv3fit.get_training_function(model_type)
    model = train(
        input_variables, output_variables, hyperparameters, train_batches, val_batches,
    )
    data_array = sample_func()
    test_dataset = xr.Dataset(data_vars={"var_in": data_array, "var_out": data_array})
    return model, test_dataset


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
    model, test_dataset = train_identity_model(
        model_type, sample_func=sample_func, hyperparameters=hyperparameters
    )
    out_dataset = model.predict(test_dataset)
    rmse = np.mean((out_dataset["var_out"] - test_dataset["var_out"]) ** 2) ** 0.5
    assert rmse < max_rmse
    if model_type in SYSTEM_DEPENDENT_TYPES:
        print(f"{model_type} is system dependent, not checking against regtest output")
        regtest = None
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
    fv3fit.set_random_seed(1)
    # don't set n_feature too high for this, because of curse of dimensionality
    n_sample, n_feature = int(5e3), 2
    hyperparameters = get_hyperparameter_class(model_type)()
    sample_func = get_uniform_sample_func(size=(n_sample, n_feature))

    assert_can_learn_identity(
        model_type,
        hyperparameters=hyperparameters,
        sample_func=sample_func,
        max_rmse=0.05,
        regtest=regtest,
    )


def test_train_with_same_seed_gives_same_result(model_type):
    hyperparameters = get_hyperparameter_class(model_type)()
    n_sample, n_feature = 500, 2
    fv3fit.set_random_seed(0)

    sample_func = get_uniform_sample_func(size=(n_sample, n_feature))
    first_model, test_dataset = train_identity_model(
        model_type, sample_func, hyperparameters
    )
    fv3fit.set_random_seed(0)
    sample_func = get_uniform_sample_func(size=(n_sample, n_feature))
    second_model, second_test_dataset = train_identity_model(
        model_type, sample_func, hyperparameters
    )
    xr.testing.assert_equal(test_dataset, second_test_dataset)
    first_output = first_model.predict(test_dataset)
    second_output = second_model.predict(test_dataset)
    xr.testing.assert_equal(first_output, second_output)


def test_predict_does_not_mutate_input(model_type):
    n_sample, n_feature = 100, 2
    sample_func = get_uniform_sample_func(size=(n_sample, n_feature))
    hyperparameters = get_hyperparameter_class(model_type)()
    model, test_dataset = train_identity_model(
        model_type, sample_func=sample_func, hyperparameters=hyperparameters
    )
    hash_before_predict = vcm.testing.checksum_dataarray_mapping(test_dataset)
    _ = model.predict(test_dataset)
    assert (
        vcm.testing.checksum_dataarray_mapping(test_dataset) == hash_before_predict
    ), "predict should not mutate its input"


def get_uniform_sample_func(size, low=0, high=1, seed=0):
    random = np.random.RandomState(seed=seed)

    def sample_func():
        return xr.DataArray(
            random.uniform(low=low, high=high, size=size),
            dims=["sample", "feature_dim"],
            coords=[range(size[0]), range(size[1])],
        )

    return sample_func


def test_train_default_model_on_nonstandard_identity(model_type):
    """
    The model with default configuration options can learn the identity function,
    using gaussian-sampled data around a non-zero value with non-unit variance.
    """
    low, high = 100, 200
    # don't set n_feature too high for this, because of curse of dimensionality
    n_sample, n_feature = int(5e3), 2
    hyperparameters = get_hyperparameter_class(model_type)()
    sample_func = get_uniform_sample_func(
        low=low, high=high, size=(n_sample, n_feature)
    )

    assert_can_learn_identity(
        model_type,
        hyperparameters=hyperparameters,
        sample_func=sample_func,
        max_rmse=0.05 * (high - low),
    )


def test_dump_and_load_default_maintains_prediction(model_type):
    n_sample, n_feature = 500, 2
    sample_func = get_uniform_sample_func(size=(n_sample, n_feature))
    hyperparameters = get_hyperparameter_class(model_type)()
    model, test_dataset = train_identity_model(
        model_type, sample_func=sample_func, hyperparameters=hyperparameters
    )

    original_result = model.predict(test_dataset)
    with tempfile.TemporaryDirectory() as tmpdir:
        model.dump(tmpdir)
        loaded_model = model.__class__.load(tmpdir)
    loaded_result = loaded_model.predict(test_dataset)
    xr.testing.assert_equal(loaded_result, original_result)


def test_train_predict_multiple_stacked_dims(model_type):
    hyperparameters = get_hyperparameter_class(model_type)()
    da = xr.DataArray(np.full(fill_value=1.0, shape=(5, 10, 15)), dims=["x", "y", "z"],)
    train_dataset = xr.Dataset(
        data_vars={"var_in_0": da, "var_in_1": da, "var_out_0": da, "var_out_1": da}
    )
    train_batches = [train_dataset for _ in range(2)]
    val_batches = []
    train = fv3fit.get_training_function(model_type)
    model = train(
        ["var_in_0", "var_in_1"],
        ["var_out_0", "var_out_1"],
        hyperparameters,
        train_batches,
        val_batches,
    )
    model.predict(train_dataset)
