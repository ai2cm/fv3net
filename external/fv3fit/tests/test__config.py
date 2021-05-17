import dataclasses
import os
import tempfile
from fv3fit._shared.config import (
    _ModelTrainingConfig as ModelTrainingConfig,
    DataConfig,
    TrainingConfig,
)
import yaml

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
def test_safe_dump_dataclass_config(config):
    """
    Test that dataclass.asdict and pyyaml can be used to save the configuration class.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = os.path.join(tmpdir, "config.yaml")
        with open(filename, "w") as f:
            as_dict = dataclasses.asdict(config)
            yaml.safe_dump(as_dict, f)
        from_dict = config.__class__(**as_dict)
        assert config == from_dict


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
