import tempfile
import fv3fit
from fv3fit._shared.config import RandomForestHyperparameters
from fv3fit._shared.models import OutOfSampleModel
import numpy as np
import pytest
import xarray as xr
from fv3fit.tfdataset import tfdataset_from_batches
from fv3fit.sklearn._random_forest import train_random_forest

from tests.training.test_train import (
    get_dataset,
    get_uniform_sample_func,
    unstack_test_dataset,
)


@pytest.mark.parametrize("base_value, novelty_cutoff, output", [(1, -1, 0), (1, 1, 1)])
def test_out_of_sample_model(base_value, novelty_cutoff, output):
    input_variables = ["input"]
    output_variables = ["output"]
    base_model = fv3fit.testing.ConstantOutputPredictor(
        input_variables, output_variables
    )
    base_model.set_outputs(output=base_value)
    novelty_detector = fv3fit.testing.ConstantOutputNoveltyDetector(input_variables)
    oosModel = OutOfSampleModel(base_model, novelty_detector, novelty_cutoff)

    ds_in = xr.Dataset(
        data_vars={"input": xr.DataArray(np.zeros([3, 3, 5]), dims=["x", "y", "z"],)}
    )
    ds_out = oosModel.predict(ds_in)
    assert len(ds_out.data_vars) == 1
    assert "output" in ds_out.data_vars
    np.testing.assert_almost_equal(ds_out["output"].values, output)


@pytest.mark.parametrize("base_value, novelty_cutoff, output", [(1, -1, 0), (1, 1, 1)])
def test_out_of_sample_model_different_inputs(base_value, novelty_cutoff, output):
    base_input_variables = ["shared_input", "base_input"]
    novelty_input_variables = ["shared_input", "novelty_input"]
    output_variables = ["output"]
    base_model = fv3fit.testing.ConstantOutputPredictor(
        base_input_variables, output_variables
    )
    base_model.set_outputs(output=base_value)
    novelty_detector = fv3fit.testing.ConstantOutputNoveltyDetector(
        novelty_input_variables
    )
    oosModel = OutOfSampleModel(base_model, novelty_detector, novelty_cutoff)

    ds_in = xr.Dataset(
        data_vars={
            "shared_input": xr.DataArray(np.zeros([3, 3, 5]), dims=["x", "y", "z"],),
            "base_input": xr.DataArray(np.ones([3, 3]), dims=["x", "y"],),
            "novelty_input": xr.DataArray(np.ones([3, 3, 5]), dims=["x", "y", "z"],),
        }
    )
    ds_out = oosModel.predict(ds_in)
    assert len(ds_out.data_vars) == 1
    assert "output" in ds_out.data_vars
    np.testing.assert_almost_equal(ds_out["output"].values, output)


@pytest.mark.slow
def test_out_of_sample_identity_same_output_when_in_sample():
    fv3fit.set_random_seed(1)
    n_sample, n_tile, nx, ny, n_feature = 5, 6, 12, 12, 2
    sample_func = get_uniform_sample_func(size=(n_sample, n_tile, nx, ny, n_feature))
    regression_model_type = "sklearn_random_forest"
    input_variables, output_variables, train_dataset = get_dataset(
        regression_model_type, sample_func
    )
    _, _, test_dataset = get_dataset(regression_model_type, sample_func)

    hyperparameters = RandomForestHyperparameters(input_variables, output_variables)
    train_tfdataset = tfdataset_from_batches([train_dataset for _ in range(10)])
    val_tfdataset = tfdataset_from_batches([test_dataset])
    test_dataset = unstack_test_dataset(test_dataset)

    base_model = train_random_forest(hyperparameters, train_tfdataset, val_tfdataset)

    novelty_detector = fv3fit.testing.ConstantOutputNoveltyDetector(input_variables)
    never_oos_model = OutOfSampleModel(base_model, novelty_detector, 1)
    always_oos_model = OutOfSampleModel(base_model, novelty_detector, -1)

    base_output = base_model.predict(test_dataset)
    never_oos_output = never_oos_model.predict(test_dataset)
    always_oos_output = always_oos_model.predict(test_dataset)
    xr.testing.assert_equal(base_output, never_oos_output)
    for output_variable in output_variables:
        np.testing.assert_almost_equal(always_oos_output[output_variable].values, 0)


@pytest.mark.slow
def test_out_of_sample_dump_and_load_default_maintains_prediction():
    fv3fit.set_random_seed(1)
    n_sample, n_tile, nx, ny, n_feature = 5, 6, 12, 12, 2
    sample_func = get_uniform_sample_func(size=(n_sample, n_tile, nx, ny, n_feature))
    regression_model_type = "sklearn_random_forest"
    input_variables, output_variables, train_dataset = get_dataset(
        regression_model_type, sample_func
    )
    _, _, test_dataset = get_dataset(regression_model_type, sample_func)
    hyperparameters = RandomForestHyperparameters(input_variables, output_variables)
    train_tfdataset = tfdataset_from_batches([train_dataset for _ in range(10)])
    val_tfdataset = tfdataset_from_batches([test_dataset])
    test_dataset = unstack_test_dataset(test_dataset)

    base_model = train_random_forest(hyperparameters, train_tfdataset, val_tfdataset)
    novelty_detector = fv3fit.testing.ConstantOutputNoveltyDetector(input_variables)
    original_model = OutOfSampleModel(base_model, novelty_detector, 1)

    with tempfile.TemporaryDirectory() as tmpdir:
        fv3fit.dump(original_model, tmpdir)
        loaded_model = fv3fit.load(tmpdir)

    original_result = original_model.predict(test_dataset)
    loaded_result = loaded_model.predict(test_dataset)

    xr.testing.assert_allclose(loaded_result, original_result, rtol=1e-4)
