import dataclasses
import fv3fit
from fv3fit._shared.config import get_hyperparameter_class
from fv3fit._shared.hyperparameters import Hyperparameters
from fv3fit._shared.novelty_detector import NoveltyDetector
from fv3fit.tfdataset import tfdataset_from_batches
from fv3fit.sklearn._min_max_novelty_detector import MinMaxNoveltyDetector
import pytest
from typing import Callable, Sequence, Union
import xarray as xr

from tests.training.test_train import (
    get_dataset_default,
    get_uniform_sample_func,
    unstack_test_dataset,
)

# novelty detection predictors that can be tested on any input data, but whose outputs
# will not correspond to the labels of standard supervised prediction tasks
NOVELTY_TRAINING_TYPES = ["min_max_novelty_detector", "ocsvm_novelty_detector"]


@pytest.fixture(params=NOVELTY_TRAINING_TYPES)
def model_type(request):
    return request.param


@dataclasses.dataclass
class NoveltyTrainingResult:
    model: NoveltyDetector
    output_variables: Sequence[str]
    test_dataset: xr.Dataset
    hyperparameters: Hyperparameters


def train_novelty_detector(
    model_type: str,
    sample_func: Callable[[], xr.DataArray],
    hyperparameters: Hyperparameters = None,
):
    input_variables, output_variables, train_dataset = get_dataset_default(sample_func)
    if hyperparameters is None:
        cls = get_hyperparameter_class(model_type)
        hyperparameters = cls.init_testing(input_variables, output_variables)
    input_variables, _, test_dataset = get_dataset_default(sample_func)
    train_tfdataset = tfdataset_from_batches([train_dataset for _ in range(10)])
    # val_tfdataset is discarded and not used for training
    val_tfdataset = tfdataset_from_batches([test_dataset])
    train = fv3fit.get_training_function(model_type)
    model = train(hyperparameters, train_tfdataset, val_tfdataset)
    test_dataset = unstack_test_dataset(test_dataset)
    output_variables = MinMaxNoveltyDetector._NOVELTY_OUTPUT_VAR
    return NoveltyTrainingResult(model, output_variables, test_dataset, hyperparameters)


def scale_test_sample(
    test_dataset: xr.Dataset, scaling: Union[int, float] = 100
) -> xr.Dataset:
    for data_var in test_dataset.data_vars:
        test_dataset[data_var] = scaling * test_dataset[data_var]
    return test_dataset


def assert_correct_output(model_type: str, sample_func: Callable[[], xr.DataArray]):
    """
    Args:
        model_type: type of model to train
        sample_func: function that returns example DataArrays for training and
            validation, should return different data on subsequent calls
    """
    result = train_novelty_detector(model_type, sample_func)
    out_dataset = result.model.predict_novelties(result.test_dataset)
    # dimensions are the same, except for "z"
    output_dimensions = set(out_dataset.dims.keys())
    output_dimensions.add("z")
    test_dataset_dimensions = set(result.test_dataset.dims.keys())
    assert output_dimensions == test_dataset_dimensions
    # output is is_novelty
    assert set(out_dataset.data_vars.keys()) == set(
        [NoveltyDetector._NOVELTY_OUTPUT_VAR, NoveltyDetector._SCORE_OUTPUT_VAR]
    )
    # outputs are either 0 or 1 (and at least one output is not a novelty)
    assert out_dataset[NoveltyDetector._NOVELTY_OUTPUT_VAR].max() <= 1
    assert out_dataset[NoveltyDetector._NOVELTY_OUTPUT_VAR].min() == 0

    if model_type == "min_max_novelty_detector":
        assert out_dataset[NoveltyDetector._SCORE_OUTPUT_VAR].min() == 0
    elif model_type == "ocsvm_novelty_detector":
        assert out_dataset[NoveltyDetector._SCORE_OUTPUT_VAR].max() < 0


def assert_extreme_novelties(
    model_type: str,
    sample_func: Callable[[], xr.DataArray],
    scaling: Union[int, float] = 100,
    epsilon: float = 0.01,
):
    """
    Args:
        model_type: type of model to train
        sample_func: function that returns example DataArrays for training and
            validation, should return different data on subsequent calls
        scaling: multiplicative scaling applied to force a sample to be an outlier
        epsilon: acceptable percentage of false negatives
    """
    result = train_novelty_detector(model_type, sample_func)
    scaled_test_dataset = scale_test_sample(result.test_dataset, scaling=scaling)
    out_dataset = result.model.predict_novelties(scaled_test_dataset)
    # almost every output is a novelty
    assert out_dataset[NoveltyDetector._NOVELTY_OUTPUT_VAR].mean() >= 1 - epsilon
    assert out_dataset[NoveltyDetector._NOVELTY_OUTPUT_VAR].mean() > 0


@pytest.mark.slow
def test_train_novelty_default_correct_output(model_type: str, regtest):
    """
    The model has properly formatted output, including (1) thesame dimensions as the
    input (besides the vertical dimension), (2) the correct data variable, and
    (3) outputs in the correct [0, 1] range.
    """
    n_sample, n_tile, nx, ny, n_feature = 10, 6, 12, 12, 2
    sample_func = get_uniform_sample_func(size=(n_sample, n_tile, nx, ny, n_feature))
    assert_correct_output(model_type, sample_func)


@pytest.mark.slow
def test_train_novelty_default_extreme_novelties(model_type: str, regtest):
    """
    When testing coordinates are scaled by a very large amount, nearly every sample is
    deemed a novelty.
    """
    n_sample, n_tile, nx, ny, n_feature = 10, 6, 12, 12, 2
    sample_func = get_uniform_sample_func(size=(n_sample, n_tile, nx, ny, n_feature))
    assert_extreme_novelties(model_type, sample_func)
