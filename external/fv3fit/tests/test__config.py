import os
import tempfile
from fv3fit._shared.config import (
    _ModelTrainingConfig as ModelTrainingConfig,
    DataConfig,
    TrainingConfig,
)

import pytest


legacy_config = ModelTrainingConfig(
    model_type="great_model",
    hyperparameters={"max_depth": 10},
    input_variables=["in0", "in1"],
    output_variables=["out0, out1"],
    batch_function="batches_from_mapper",
    batch_kwargs={"timesteps_per_batch": 1},
)


def test_dump_and_load_legacy_config():
    with tempfile.TemporaryDirectory() as tmpdir:
        legacy_config.dump(tmpdir)
        loaded = ModelTrainingConfig.load(os.path.join(tmpdir, "training_config.yml"))
        assert legacy_config.asdict() == loaded.asdict()


data_config = DataConfig(
    variables=["a", "b"],
    data_path="/my/path",
    batch_function="batch_func",
    batch_kwargs={"key": "value"},
)

training_config = TrainingConfig(
    model_type="DummyModel", input_variables=["a"], output_variables=["b"]
)


@pytest.mark.parametrize("config", [data_config, training_config])
def test_dump_and_load_config(config):
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = os.path.join(tmpdir, "config.yml")
        config.dump(filename)
        loaded = config.__class__.load(filename)
        assert config == loaded


@pytest.mark.parametrize(
    "scaler_type, expected",
    [("mass", ["pressure_thickness_of_atmospheric_layer"]), (None, [])],
)
def test_config_variables(scaler_type, expected):
    config = ModelTrainingConfig(
        model_type="great_model",
        hyperparameters={"max_depth": 10},
        input_variables=["in0", "in1"],
        output_variables=["out0, out1"],
        batch_function="batches_from_mapper",
        batch_kwargs={"timesteps_per_batch": 1},
        scaler_type=scaler_type,
    )
    assert expected == config.additional_variables
