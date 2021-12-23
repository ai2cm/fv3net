import argparse
import dacite
import dataclasses
from fv3fit import (
    OptimizerConfig,
    LearningRateScheduleConfig,
    TrainingConfig,
    DenseHyperparameters,
)
import os
import tempfile
import yaml
from fv3fit._shared.config import (
    to_nested_dict,
    _to_flat_dict,
    get_arg_updated_config_dict,
    _add_items_to_parser_arguments,
)

import pytest


@pytest.mark.parametrize("hyperparameters", [{}])
def test_dense_training_config_uses_optimizer_config(hyperparameters):
    config_dict = {
        "model_type": "dense",
        "input_variables": [],
        "output_variables": [],
        "hyperparameters": hyperparameters,
    }
    training_config = TrainingConfig.from_dict(config_dict)
    assert isinstance(training_config.hyperparameters.optimizer_config, OptimizerConfig)


def _get_exponential_decay():
    return LearningRateScheduleConfig(
        name="ExponentialDecay",
        kwargs=dict(initial_learning_rate=1e-4, decay_steps=100, decay_rate=0.95,),
    )


def test_OptimizerConfig_learning_rate_error_on_dual_specify():

    with pytest.raises(ValueError):
        OptimizerConfig(
            name="Adam",
            kwargs=dict(learning_rate=1e-5),
            learning_rate_schedule=_get_exponential_decay(),
        )


def test_OptimizerConfig_learning_rate():

    assert OptimizerConfig(
        name="Adam", learning_rate_schedule=_get_exponential_decay(),
    ).instance


def test_safe_dump_training_config():
    """
    Test that dataclass.asdict and pyyaml can be used to save the configuration class,
    and that the relationship between model_type and hyperparameter class is
    preserved when restoring configuration using TrainingConfig.from_dict.
    """
    # TODO: extend this test to run not just for Dense, but for all registered models
    config = TrainingConfig(
        model_type="dense",  # an arbitrary model type
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
        "model_type": "dense",
        "input_variables": [],
        "output_variables": [],
        "hyperparameters": hyperparameters,
    }
    if passes:
        TrainingConfig.from_dict(config_dict)
    else:
        with pytest.raises(dacite.exceptions.UnexpectedDataError):
            TrainingConfig.from_dict(config_dict)


def get_cfg_and_args_dicts():

    config_d = {
        "top": 1,
        "seq": [dict(a=1), dict(a=2)],
        "nested": {"k1": 2, "k2": 3, "double_nest": {"k1": 4, "k2": 5}},
    }

    flat_d = {
        "top": 1,
        "seq": [dict(a=1), dict(a=2)],
        "nested.k1": 2,
        "nested.k2": 3,
        "nested.double_nest.k1": 4,
        "nested.double_nest.k2": 5,
    }

    return config_d, flat_d


def test_to_flat_dict():

    config_d, expected = get_cfg_and_args_dicts()
    result = _to_flat_dict(config_d)
    assert result == expected


def test_to_nested_dict():

    expected, args_d = get_cfg_and_args_dicts()
    result = to_nested_dict(args_d)
    assert result == expected


def test_flat_dict_round_trip():

    config_d, _ = get_cfg_and_args_dicts()

    args_d = _to_flat_dict(config_d)
    result = to_nested_dict(args_d)

    assert result == config_d


def test__add_items_to_parser_args_no_seq():

    d = {
        "number": 1.0,
        "string.nested": "hi",
        "boolean": False,
    }

    parser = argparse.ArgumentParser()
    _add_items_to_parser_arguments(d, parser)

    default = parser.parse_args([])
    for k, v in d.items():
        parsed = vars(default)
        assert parsed[k] == v

    args = ["--number", "2.0", "--string.nested", "there"]
    specified = vars(parser.parse_args(args))
    assert specified["number"] == 2.0
    assert specified["string.nested"] == "there"


def test__add_items_to_parser_args_mapping_fail():

    d = {"mapping": {}}

    parser = argparse.ArgumentParser()
    with pytest.raises(ValueError):
        _add_items_to_parser_arguments(d, parser)


@pytest.mark.parametrize(
    "args, expected",
    [
        (["--boolean", "True"], True),
        (["--boolean", "true"], True),
        (["--boolean", "false"], False),
        ([], False),
    ],
)
def test__add_items_to_parser_args_mapping_bools(args, expected):

    d = {"boolean": False}

    parser = argparse.ArgumentParser()
    _add_items_to_parser_arguments(d, parser)

    assert parser.parse_args(args).boolean == expected


@pytest.mark.parametrize(
    "args, expected_seq",
    [
        ([], [1, 2, 3]),
        (["--seq"], []),
        (["--seq", "1"], ["1"]),
        (["--seq", "1", "2"], ["1", "2"]),
    ],
    ids=["default", "empty", "single", "multiple"],
)
def test__add_items_to_parser_args_seq(args, expected_seq):

    d = {"seq": [1, 2, 3]}

    parser = argparse.ArgumentParser()
    _add_items_to_parser_arguments(d, parser)

    parsed = parser.parse_args(args)
    assert parsed.seq == expected_seq


def test__add_items_to_parser_args_mapping_error():

    d = {"mapping": dict(a=1)}
    parser = argparse.ArgumentParser()

    with pytest.raises(ValueError):
        _add_items_to_parser_arguments(d, parser)


def test_get_updated_config_dict():

    defaults = {
        "epochs": 1,
        "model": {"architecture": {"name": "linear"}},
        "transform": {"input_variables": ["field"]},
        "unchanged": "same",
    }

    arg_updates = [
        "--epochs",
        "4",
        "--model.architecture.name",
        "rnn",
        "--transform.input_variables",
        "A",
        "B",
        "C",
    ]

    updated = get_arg_updated_config_dict(arg_updates, defaults)

    assert updated["epochs"] == 4
    assert updated["model"]["architecture"]["name"] == "rnn"
    assert updated["transform"]["input_variables"] == ["A", "B", "C"]
    assert updated["unchanged"] == "same"
