import xarray as xr
import numpy as np
import fv3fit
from scream_run.steppers.machine_learning import (
    MultiModelAdapter,
    open_model,
    predict,
    MachineLearningConfig,
)
import pytest


def _model_dataset() -> xr.Dataset:
    nz = 128
    arr = np.zeros((1, nz))
    dims = ["ncol", "z"]

    data = xr.Dataset({"qv": (dims, arr), "T_mid": (dims, arr),})  # noqa: E231

    return data


def test_sample_scream_ml_config():
    config = MachineLearningConfig(
        models=["gs://vcm-ml-experiments/scream-n2f/test-scream-train-model"]
    )
    model = open_model(config)
    assert model.input_variables == {"qv", "T_mid", "cos_zenith_angle"}


def test_mock_scream_ml_prediction():
    data = _model_dataset()
    nz = data.sizes["z"]
    output_variables = ["dQ1", "dQ2"]
    outputs = {
        "dQ1": np.full(nz, 0.0),
        "dQ2": np.full(nz, 0.0),
    }
    predictor = fv3fit.testing.ConstantOutputPredictor(
        input_variables=["T_mid", "qv"], output_variables=output_variables,
    )
    predictor.set_outputs(**outputs)
    model = MultiModelAdapter([predictor])
    output = predict(model, data, dt=1.0)
    assert output["dQ1"].values.all() == pytest.approx(0.0)
    assert output["dQ2"].values.all() == pytest.approx(0.0)
    assert output["dQ1"].dims == ("ncol", "z")
    assert output["dQ2"].dims == ("ncol", "z")
    assert output["dQ1"].sizes["z"] == nz
    assert output["dQ2"].sizes["z"] == nz
    assert predictor.input_variables == ["T_mid", "qv"]
    assert predictor.output_variables == ["dQ1", "dQ2"]
