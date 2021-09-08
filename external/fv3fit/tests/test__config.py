import dacite
import dataclasses
from fv3fit import DenseHyperparameters, OptimizerConfig, TrainingConfig
import os
import tempfile
import yaml

import pytest


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


def test_safe_dump_training_config():
    """
    Test that dataclass.asdict and pyyaml can be used to save the configuration class,
    and that the relationship between model_type and hyperparameter class is
    preserved when restoring configuration using TrainingConfig.from_dict.
    """
    # TODO: extend this test to run not just for Dense, but for all registered models
    config = TrainingConfig(
        model_type="DenseModel",  # an arbitrary model type
        hyperparameters=DenseHyperparameters(
            input_variables=["a"], output_variables=["b"],
        ),
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = os.path.join(tmpdir, "config.yaml")
        with open(filename, "w") as f:
            as_dict = dataclasses.asdict(config)
            yaml.safe_dump(as_dict, f)
        from_dict = TrainingConfig.from_dict(as_dict)
        assert config == from_dict


@pytest.mark.parametrize(
    "hyperparameters, passes",
    [
        pytest.param(
            {"dense_network": {"width": 32}}, True, id="pass_has_DenseModelConfig"
        ),
        pytest.param(
            {"training_loop": {"epochs": 2}}, True, id="pass_has_TrainingLoopConfig"
        ),
        pytest.param(
            {"width": 32}, False, id="fail_has_DenseModelConfig_param_in_top_level"
        ),
        pytest.param(
            {"epochs": 2}, False, id="fail_has_TrainingLoopConfig_param_in_top_level"
        ),
    ],
)
def test__load_config_catches_errors_with_strict_checking(hyperparameters, passes):
    config_dict = {
        "model_type": "DenseModel",
        "input_variables": [],
        "output_variables": [],
        "hyperparameters": hyperparameters,
    }
    if passes:
        TrainingConfig.from_dict(config_dict)
    else:
        with pytest.raises(dacite.exceptions.UnexpectedDataError):
            TrainingConfig.from_dict(config_dict)
