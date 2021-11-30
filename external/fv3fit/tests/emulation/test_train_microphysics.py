import pytest
import sys
import yaml
from dataclasses import asdict

from fv3fit._shared.config import _to_flat_dict
from fv3fit.emulation.data.config import TransformConfig

from fv3fit.train_microphysics import (
    TrainConfig,
    MicrophysicsConfig,
    _get_out_samples,
    get_default_config,
    main,
)


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

    d = asdict(config)
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
    result = TrainConfig.from_dict(asdict(expected))

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
    flat_dict = _to_flat_dict(asdict(expected))
    result = TrainConfig.from_flat_dict(flat_dict)
    assert result == expected


def test_TrainConfig_from_yaml(tmp_path):

    default = get_default_config()

    yaml_path = str(tmp_path / "train_config.yaml")
    with open(yaml_path, "w") as f:
        yaml.safe_dump(asdict(default), f)

        loaded = TrainConfig.from_yaml_path(yaml_path)

        assert loaded == default


def test_TrainConfig_from_args_default():

    default = get_default_config()

    args = ["--config-path", "default"]
    config = TrainConfig.from_args(args=args)

    assert config == default


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

    config_dict = asdict(get_default_config())
    config_dict["out_url"] = str(tmp_path)
    config_dict["use_wandb"] = False
    config_dict["nfiles"] = 4
    config_dict["nfiles_valid"] = 4
    config_dict["epochs"] = 1

    config = TrainConfig.from_dict(config_dict)

    main(config)
