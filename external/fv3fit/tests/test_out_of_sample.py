import tempfile

import fsspec
import yaml
import fv3fit
from fv3fit._shared.config import RandomForestHyperparameters
from fv3fit._shared.models import OutOfSampleModel
import numpy as np
import os
import pytest
import xarray as xr
from fv3fit._shared.novelty_detector import NoveltyDetector
from fv3fit._shared.taper_function import (
    ExponentialDecayTaperFunction,
    MaskTaperFunction,
    RampTaperFunction,
)
from fv3fit.testing import ConstantOutputPredictor
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
    assert len(ds_out.data_vars) == 4
    # Verifies that the default mask taper function works properly
    np.testing.assert_allclose(
        ds_out[NoveltyDetector._NOVELTY_OUTPUT_VAR],
        1 - ds_out[OutOfSampleModel._TAPER_VALUES_OUTPUT_VAR],
    )
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
    assert len(ds_out.data_vars) == 4
    np.testing.assert_allclose(
        ds_out[NoveltyDetector._NOVELTY_OUTPUT_VAR],
        1 - ds_out[OutOfSampleModel._TAPER_VALUES_OUTPUT_VAR],
    )
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
    for output_variable in output_variables:
        xr.testing.assert_equal(
            base_output[output_variable], never_oos_output[output_variable]
        )
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
    cutoff = 1
    original_model = OutOfSampleModel(base_model, novelty_detector, cutoff)

    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = os.path.join(tmpdir, "base")
        novelty_path = os.path.join(tmpdir, "novelty")
        fv3fit.dump(base_model, base_path)
        fv3fit.dump(novelty_detector, novelty_path)

        options = {
            "base_model_path": base_path,
            "novelty_detector_path": novelty_path,
            "cutoff": cutoff,
        }
        with fsspec.open(
            os.path.join(tmpdir, OutOfSampleModel._CONFIG_FILENAME), "w"
        ) as f:
            yaml.safe_dump(options, f)

        loaded_model = OutOfSampleModel.load(tmpdir)

    original_result = original_model.predict(test_dataset)
    loaded_result = loaded_model.predict(test_dataset)

    xr.testing.assert_allclose(loaded_result, original_result, rtol=1e-4)


@pytest.mark.slow
def test_masked_tapering():
    cutoff = 3
    taper = MaskTaperFunction(cutoff)
    novelty_score = xr.DataArray([[1, 3, 5], [6, 4, 2]])
    taper_values = taper.get_taper_value(novelty_score)
    np.testing.assert_almost_equal(taper_values, np.asarray([[1, 1, 0], [0, 0, 1]]))


@pytest.mark.slow
def test_ramp_tapering():
    ramp_min = 2
    ramp_max = 5
    taper = RampTaperFunction(ramp_min, ramp_max)
    novelty_score = xr.DataArray([[1, 3, 5], [6, 4, 2]])
    taper_values = taper.get_taper_value(novelty_score)
    np.testing.assert_almost_equal(
        taper_values, np.asarray([[1, 2 / 3, 0], [0, 1 / 3, 1]])
    )


@pytest.mark.slow
def test_decay_tapering():
    threshold = 2
    rate = 0.5
    taper = ExponentialDecayTaperFunction(threshold, rate)
    novelty_score = xr.DataArray([[1, 3, 5], [6, 4, 2]])
    taper_values = taper.get_taper_value(novelty_score)
    np.testing.assert_almost_equal(
        taper_values, np.asarray([[1, 2 ** -1, 2 ** -3], [2 ** -4, 2 ** -2, 1]])
    )


def load_oos_model_with_tapering_config(tapering_config):
    base_model = ConstantOutputPredictor([], [])
    novelty_detector = fv3fit.testing.ConstantOutputNoveltyDetector([])

    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = os.path.join(tmpdir, "base")
        novelty_path = os.path.join(tmpdir, "novelty")
        fv3fit.dump(base_model, base_path)
        fv3fit.dump(novelty_detector, novelty_path)

        config = {
            "base_model_path": base_path,
            "novelty_detector_path": novelty_path,
        }
        if tapering_config is not None:
            config["tapering_function"] = tapering_config

        with fsspec.open(
            os.path.join(tmpdir, OutOfSampleModel._CONFIG_FILENAME), "w"
        ) as f:
            yaml.safe_dump(config, f)
        loaded_model = OutOfSampleModel.load(tmpdir)
        return loaded_model


@pytest.mark.slow
def test_taper_loading_default_config():
    loaded_model = load_oos_model_with_tapering_config(None)
    assert isinstance(loaded_model.taper, MaskTaperFunction)
    assert loaded_model.taper.cutoff == 0


@pytest.mark.slow
def test_taper_loader_default_mask_config():
    tapering_config = {"name": MaskTaperFunction.__name__}
    loaded_model = load_oos_model_with_tapering_config(tapering_config)
    assert isinstance(loaded_model.taper, MaskTaperFunction)
    assert loaded_model.taper.cutoff == 0


@pytest.mark.slow
def test_taper_loader_mask_config():
    tapering_config = {"name": MaskTaperFunction.__name__, "cutoff": 3}
    loaded_model = load_oos_model_with_tapering_config(tapering_config)
    assert isinstance(loaded_model.taper, MaskTaperFunction)
    assert loaded_model.taper.cutoff == 3


@pytest.mark.slow
def test_taper_loader_default_ramp_config():
    tapering_config = {"name": RampTaperFunction.__name__}
    loaded_model = load_oos_model_with_tapering_config(tapering_config)
    assert isinstance(loaded_model.taper, RampTaperFunction)
    assert loaded_model.taper.ramp_min == 0
    assert loaded_model.taper.ramp_max == 1


@pytest.mark.slow
def test_taper_loader_ramp_config():
    tapering_config = {
        "name": RampTaperFunction.__name__,
        "ramp_min": -1,
        "ramp_max": 2,
    }
    loaded_model = load_oos_model_with_tapering_config(tapering_config)
    assert isinstance(loaded_model.taper, RampTaperFunction)
    assert loaded_model.taper.ramp_min == -1
    assert loaded_model.taper.ramp_max == 2


@pytest.mark.slow
def test_taper_loader_default_decay_config():
    tapering_config = {"name": ExponentialDecayTaperFunction.__name__}
    loaded_model = load_oos_model_with_tapering_config(tapering_config)
    assert isinstance(loaded_model.taper, ExponentialDecayTaperFunction)
    assert loaded_model.taper.threshold == 0
    assert loaded_model.taper.rate == 0.5


@pytest.mark.slow
def test_taper_loader_decay_config():
    tapering_config = {
        "name": ExponentialDecayTaperFunction.__name__,
        "threshold": 1,
        "rate": 1 / 3,
    }
    loaded_model = load_oos_model_with_tapering_config(tapering_config)
    assert isinstance(loaded_model.taper, ExponentialDecayTaperFunction)
    assert loaded_model.taper.threshold == 1
    assert loaded_model.taper.rate == 1 / 3
