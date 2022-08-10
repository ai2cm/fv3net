import dataclasses
import numpy as np
import tensorflow as tf
import xarray as xr
import fv3fit
from typing import Optional, Sequence, TextIO
from fv3fit.pytorch.predict import PytorchModel
import pytest
from fv3fit.pytorch.graph.graph import GraphHyperparameters

GENERAL_TRAINING_TYPES = [
    "graph",
]

# automatically test on every registered training class
@pytest.fixture(params=GENERAL_TRAINING_TYPES)
def model_type(request):
    return request.param


@dataclasses.dataclass
class TrainingResult:
    model: PytorchModel
    output_variables: Sequence[str]
    test_dataset: tf.data.Dataset
    hyperparameters: GraphHyperparameters


def train_identity_model(hyperparameters=None):
    time, grid, nz = 50, 6 * 6 * 6, 2
    low, high = 0.0, 1.0
    np.random.seed(0)
    input_variable, output_variables, train_dataset = get_data(
        size=(time, grid, nz), low=low - 1.0, high=high + 1.0
    )
    np.random.seed(1)
    _, _, val_tfdataset = get_data(size=(time, grid, nz), low=low, high=high)
    np.random.seed(2)
    sample_test = get_uniform_sample_func(size=(grid, nz), low=low, high=high)
    test_dataset = xr.Dataset({"a": sample_test()})
    hyperparameters = GraphHyperparameters(input_variable, output_variables)
    train = fv3fit.get_training_function("graph")
    model = train(hyperparameters, train_dataset, val_tfdataset)
    return TrainingResult(model, output_variables, test_dataset, hyperparameters)


@pytest.mark.slow
def test_train_default_model_on_identity(regtest):
    """
    The model with default configuration options can learn the identity function,
    using gaussian-sampled data around 0 with unit variance.
    """
    assert_can_learn_identity(
        max_rmse=0.5, regtest=regtest,
    )


def assert_can_learn_identity(
    max_rmse: float, regtest: Optional[TextIO] = None,
):
    """
    Args:
        model_type: type of model to train
        hyperparameters: model configuration
        max_rmse: maximum permissible root mean squared error
        regtest: if given, write hash of output dataset to this file object
    """
    result = train_identity_model()
    out_dataset = result.model.predict(result.test_dataset)
    for name in result.output_variables:
        assert out_dataset[name].dims == result.test_dataset[name].dims
    rmse = (
        np.mean(
            [
                np.mean((out_dataset[name] - result.test_dataset[name]) ** 2)
                / np.std(result.test_dataset[name]) ** 2
                for name in result.output_variables
            ]
        )
        ** 0.5
    )
    assert rmse < max_rmse


def get_data(size, low=0, high=1) -> tf.data.Dataset:
    n_records = 10
    input_variables = ["a"]
    output_variables = ["a"]

    def records():
        for _ in range(n_records):
            record = {
                "a": np.random.uniform(low, high, size),
            }
            yield record

    tfdataset = tf.data.Dataset.from_generator(
        records,
        output_signature={
            key: tf.TensorSpec(val.shape, dtype=val.dtype)
            for key, val in next(iter(records())).items()
        },
    )
    return input_variables, output_variables, tfdataset


def get_uniform_sample_func(size, low=0, high=1, seed=0):
    random = np.random.RandomState(seed=seed)

    def sample_func():
        return xr.DataArray(
            random.uniform(low=low, high=high, size=size),
            dims=["grid", "z"],
            coords=[range(size[i]) for i in range(len(size))],
        )

    return sample_func
