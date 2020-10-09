"""
External models don't get trained, but this test is here so we can
use the `training_batches` fixture.
"""
from typing import Sequence, Iterable
import fv3fit
import xarray as xr
import tempfile
from test_train_keras import validate_dataset_result
import pytest
import tensorflow as tf

SAMPLE_DIM_NAME = "sample"


@pytest.fixture
def model_type():
    return "ExternalModel"


@pytest.fixture
def hyperparameters():
    return {}


@pytest.fixture()
def model(
    input_variables: Iterable[str],
    output_variables: Iterable[str],
    training_batches: Sequence[xr.Dataset],
    request,
):
    batch_dataset = training_batches[0]
    X_packer = fv3fit.ArrayPacker(SAMPLE_DIM_NAME, input_variables)
    y_packer = fv3fit.ArrayPacker(SAMPLE_DIM_NAME, output_variables)
    X = X_packer.to_array(batch_dataset)
    y = y_packer.to_array(batch_dataset)
    n_input = X.shape[-1]
    n_output = y.shape[-1]
    units = 3
    inputs = tf.keras.layers.Input(n_input)
    x = tf.keras.layers.Dense(units, activation="relu")(inputs)
    outputs = tf.keras.layers.Dense(n_output)(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(loss="mse")
    return fv3fit.keras.ExternalModel(
        SAMPLE_DIM_NAME, input_variables, output_variables, model, X_packer, y_packer
    )


def test_dump_and_load_maintains_prediction(
    model: fv3fit.Predictor,
    training_batches: Sequence[xr.Dataset],
    output_variables: Iterable[str],
):
    with tempfile.TemporaryDirectory() as tmpdir:
        model.dump(tmpdir)
        loaded_model = model.__class__.load(tmpdir)
    batch_dataset = training_batches[0]
    loaded_result = loaded_model.predict(batch_dataset)
    validate_dataset_result(loaded_result, batch_dataset, output_variables)
    original_result = model.predict(batch_dataset)
    xr.testing.assert_equal(loaded_result, original_result)
