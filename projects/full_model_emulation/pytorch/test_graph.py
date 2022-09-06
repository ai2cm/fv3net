import dataclasses
import numpy as np
import tensorflow as tf
import xarray as xr
import fv3fit
from fv3fit._shared.config import TRAINING_FUNCTIONS, get_hyperparameter_class
from fv3fit.tfdataset import tfdataset_from_batches
import fv3fit
from typing import Any, Callable, Optional, Sequence, TextIO, Tuple
from fv3fit._shared.config import TRAINING_FUNCTIONS, get_hyperparameter_class
from fv3fit._shared.hyperparameters import Hyperparameters
from graphPredict import PytorchModel
from fv3fit.tfdataset import tfdataset_from_batches
import pytest
from typing import Callable, Sequence, Union
import xarray as xr


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
    test_dataset: xr.Dataset
    hyperparameters: Hyperparameters


def train_identity_model(model_type, sample_func, hyperparameters=None):
    input_variables, output_variables, train_dataset = get_data()
    if hyperparameters is None:
        cls = get_hyperparameter_class(model_type)
        hyperparameters = cls.init_testing(input_variables, output_variables)
    input_variables, output_variables, test_dataset = get_data()
    train_tfdataset = tfdataset_from_batches([train_dataset for _ in range(10)])
    val_tfdataset = tfdataset_from_batches([test_dataset])
    train = fv3fit.get_training_function(model_type)
    model = train(hyperparameters, train_tfdataset, val_tfdataset)
    return TrainingResult(model, output_variables, test_dataset, hyperparameters)



@pytest.mark.slow
def test_train_default_model_on_identity(model_type: str, regtest):
    """
    The model with default configuration options can learn the identity function,
    using gaussian-sampled data around 0 with unit variance.
    """

    fv3fit.set_random_seed(1)
    # don't set n_feature too high for this, because of curse of dimensionality
    n_sample, grid, n_feature = 5, 864 , 2
    sample_func = get_uniform_sample_func(size=(n_sample, grid, n_feature))

    assert_can_learn_identity(
        model_type, sample_func=sample_func, max_rmse=0.2, regtest=regtest,
    )

def get_uniform_sample_func(size, low=0, high=1, seed=0):
    random = np.random.RandomState(seed=seed)

    def sample_func():
        return xr.DataArray(
            random.uniform(low=low, high=high, size=size),
            dims=["sample", "grid", "z"],
            coords=[range(size[i]) for i in range(len(size))],
        )

    return sample_func


def assert_can_learn_identity(
    model_type,
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
    result = train_identity_model(model_type, sample_func=sample_func)
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



def get_data() -> tf.data.Dataset:
    n_records = 10
    nt, grid,nz = 40, 10,2
    low=0
    high=1
    
    def records():
        for _ in range(n_records):
            record = {
                "a": np.random.uniform(low, high,[ nt, grid, nz]),
                "f": np.random.uniform(low, high,[ nt, grid, nz]),
            }
            yield record
        
    tfdataset = tf.data.Dataset.from_generator(
        records,
        output_types=(tf.float32),
        output_shapes=(tf.TensorShape([nt,grid,nz])),
    )
    return tfdataset

    # set random seed
    # prepare randomly-generated "identity function" data
    # train the model
    # get the outputs from the model
    # check the outputs are sufficiently accurate