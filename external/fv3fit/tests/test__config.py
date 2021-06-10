import dataclasses
from fv3fit._shared.config import DenseHyperparameters, register_training_function
from fv3fit._shared.config import OptimizerConfig
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


@pytest.mark.parametrize("hyperparameters", [{}])
def test_dense_training_config_uses_optimizer_config(hyperparameters):
    config_dict = {
        "model_type": "DenseModel",
        "input_variables": [],
        "output_variables": [],
        "hyperparameters": hyperparameters,
    }
    training_config = TrainingConfig.from_dict(config_dict)
    assert isinstance(training_config.hyperparameters.optimizer_config, OptimizerConfig)


def test_dump_and_load_legacy_config():
    with tempfile.TemporaryDirectory() as tmpdir:
        legacy_config.dump(tmpdir)
        loaded = ModelTrainingConfig.load(os.path.join(tmpdir, "training_config.yml"))
        assert legacy_config.asdict() == loaded.asdict()


def test_safe_dump_data_config():
    """
    Test that dataclass.asdict and pyyaml can be used to save DataConfig.
    """
    config = DataConfig(
        variables=["a", "b"],
        data_path="/my/path",
        batch_function="batch_func",
        batch_kwargs={"key": "value"},
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = os.path.join(tmpdir, "config.yaml")
        with open(filename, "w") as f:
            as_dict = dataclasses.asdict(config)
            yaml.safe_dump(as_dict, f)
        from_dict = DataConfig(**as_dict)
        assert config == from_dict


def test_safe_dump_training_config():
    """
    Test that dataclass.asdict and pyyaml can be used to save the configuration class,
    and that the relationship between model_type and hyperparameter class is
    preserved when restoring configuration using TrainingConfig.from_dict.
    """
    # TODO: extend this test to run not just for Dense, but for all registered models
    config = TrainingConfig(
        model_type="DenseModel",  # an arbitrary model type
        input_variables=["a"],
        output_variables=["b"],
        hyperparameters=DenseHyperparameters(),
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = os.path.join(tmpdir, "config.yaml")
        with open(filename, "w") as f:
            as_dict = dataclasses.asdict(config)
            yaml.safe_dump(as_dict, f)
        from_dict = TrainingConfig.from_dict(as_dict)
        assert config == from_dict


@pytest.mark.parametrize(
    "scaler_type, expected",
    [("mass", ["pressure_thickness_of_atmospheric_layer"]), (None, [])],
)
def test_config_additional_variables(scaler_type, expected):
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
