import argparse
import pytest
import sys
import yaml

from fv3fit.emulation.data.config import TransformConfig

from fv3fit.train_microphysics import (
    TrainConfig,
    to_flat_dict,
    to_nested_dict,
    MicrophysicsConfig,
    _get_out_samples,
    _add_items_to_parser_arguments,
    get_default_config,
    get_arg_updated_config_dict,
    main,
)


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
    result = to_flat_dict(config_d)
    assert result == expected


def test_to_nested_dict():

    expected, args_d = get_cfg_and_args_dicts()
    result = to_nested_dict(args_d)
    assert result == expected


def test_flat_dict_round_trip():

    config_d, _ = get_cfg_and_args_dicts()

    args_d = to_flat_dict(config_d)
    result = to_nested_dict(args_d)

    assert result == config_d


def test__get_out_samples():

    samples = [1, 2, 3, 4]
    names = ["B", "dC", "dD", "A"]

    config = MicrophysicsConfig(
        direct_out_variables=["A", "B"],
        residual_out_variables={"C": "C_in", "D": "D_in"},
        tendency_outputs={"C": "dC", "D": "dD"},
    )

    direct, residual = _get_out_samples(config, samples, names)

    assert direct == [4, 1]
    assert residual == [2, 3]


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
    assert specified["number"] == "2.0"
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


def test_TrainConfig_defaults():

    config = TrainConfig(
        train_url="train_path",
        test_url="test_path",
        out_url="save_path",
        transform=TransformConfig(),
        model=MicrophysicsConfig(),
    )

    assert config  # for linter


def test_get_default_config():

    config = get_default_config()
    assert isinstance(config, TrainConfig)


def test_TrainConfig_asdict():

    config = TrainConfig(
        train_url="train_path", test_url="test_path", out_url="save_path",
    )

    d = config.asdict()
    assert d["train_url"] == "train_path"
    assert d["model"]["architecture"]["name"] == "linear"


def test_TrainConfig_from_dict():

    d = dict(
        train_url="train_path",
        test_url="test_path",
        out_url="save_path",
        model=dict(architecture={"name": "rnn"}),
    )

    config = TrainConfig.from_dict(d)
    assert config.train_url == "train_path"
    assert config.model.architecture.name == "rnn"


def test_TrainConfig_from_dict_full():

    expected = get_default_config()
    result = TrainConfig.from_dict(expected.asdict())

    assert result == expected


def test_TrainConfig_from_flat_dict():

    d = {
        "train_url": "train_path",
        "test_url": "test_path",
        "out_url": "out_path",
        "model.architecture.name": "rnn",
    }

    config = TrainConfig.from_flat_dict(d)

    assert config.train_url == "train_path"
    assert config.model.architecture.name == "rnn"

    expected = get_default_config()
    result = TrainConfig.from_flat_dict(expected.as_flat_dict())
    assert result == expected


def test_TrainConfig_from_yaml(tmp_path):

    default = get_default_config()

    yaml_path = str(tmp_path / "train_config.yaml")
    with open(yaml_path, "w") as f:
        yaml.safe_dump(default.asdict(), f)

        loaded = TrainConfig.from_yaml_path(yaml_path)

        assert loaded == default


def test_TrainConfig_from_args_default():

    default = get_default_config()

    args = ["--config-path", "default"]
    config = TrainConfig.from_args(args=args)

    assert config == default


def test_get_updated_config_dict():

    defaults = get_default_config().as_flat_dict()

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

    assert updated["epochs"] == "4"
    assert updated["model.architecture.name"] == "rnn"
    assert updated["transform.input_variables"] == ["A", "B", "C"]


def test_TrainConfig_from_args_sysargv(monkeypatch):

    args = [
        "unused_sysv_arg",
        "--config-path",
        "default",
        "--epochs",
        "4",
        "--model.architecture.name",
        "rnn",
    ]
    monkeypatch.setattr(sys, "argv", args)

    config = TrainConfig.from_args()

    assert config.epochs == 4
    assert config.model.architecture.name == "rnn"


def test_TrainConfig_invalid_input_vars():

    args = [
        "--config-path",
        "default",
        "--model.input_variables",
        "A",
        "B",
        "--transform.input_variables",
        "A",
        "C",
    ]

    with pytest.raises(ValueError):
        TrainConfig.from_args(args)

    args = [
        "--config-path",
        "default",
        "--model.direct_out_variables",
        "A",
        "B",
        "--transform.output_variables",
        "A",
        "C",
    ]

    with pytest.raises(ValueError):
        TrainConfig.from_args(args)


@pytest.mark.regression
def test_training_entry_integration(tmp_path):

    config_dict = get_default_config().asdict()
    config_dict["out_url"] = str(tmp_path)
    config_dict["use_wandb"] = False
    config_dict["nfiles"] = 4
    config_dict["nfiles_valid"] = 4
    config_dict["epochs"] = 1

    config = TrainConfig.from_dict(config_dict)

    main(config)
