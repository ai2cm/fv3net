from pathlib import PosixPath
import sys
from fv3fit.dataclasses import asdict_with_enum as asdict
from unittest.mock import Mock

from fv3fit.emulation.losses import CustomLoss

import pytest
import tensorflow as tf
from fv3fit._shared.config import to_flat_dict
from fv3fit.emulation.data.config import TransformConfig
from fv3fit.emulation.layers.architecture import ArchitectureConfig
from fv3fit.emulation.models import MicrophysicsConfig
from fv3fit.emulation.zhao_carr_fields import Field
from fv3fit.train_microphysics import (
    TrainConfig,
    TransformedParameters,
    get_default_config,
    main,
)


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
        train_url="train_path",
        test_url="test_path",
        out_url="save_path",
        model=MicrophysicsConfig(),
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
    flat_dict = to_flat_dict(asdict(expected))
    result = TrainConfig.from_flat_dict(flat_dict)
    assert result == expected


def test_TrainConfig_from_yaml(tmp_path: PosixPath):
    default = TrainConfig(test_url=".", train_url=".", out_url=".")
    yaml_path = tmp_path / "train_config.yaml"
    yaml_path.write_text(default.to_yaml())
    loaded = TrainConfig.from_yaml_path(yaml_path.as_posix())
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


@pytest.mark.regression
@pytest.mark.slow
def test_training_entry_integration(tmp_path):

    config_dict = asdict(get_default_config())
    config_dict["out_url"] = str(tmp_path)
    config_dict["use_wandb"] = False
    config_dict["nfiles"] = 4
    config_dict["nfiles_valid"] = 4
    config_dict["epochs"] = 1

    config = TrainConfig.from_dict(config_dict)

    main(config)


def test_MicrophysicsHyperParameters_build_model():
    field = Field("out", "in")
    in_ = "in"
    out = "out"
    config = TransformedParameters(
        model=MicrophysicsConfig(
            input_variables=[in_],
            direct_out_variables=[out],
            architecture=ArchitectureConfig("dense"),
            timestep_increment_sec=900,
        ),
    )

    assert set(config.input_variables) == {"in"}
    data = {field.input_name: tf.ones((1, 10)), field.output_name: tf.ones((1, 10))}
    model = config.build_model(data, config.build_transform(data))
    assert field.output_name in model(data)


def test_TrainConfig_build_loss():
    config = TrainConfig(".", ".", ".", loss=CustomLoss(loss_variables=["x"]))
    # needs to be random or the normalized loss will have nan
    data = {"x": tf.random.uniform(shape=(4, 10))}
    transform = Mock()
    transform.forward.return_value = data
    loss = config.build_loss(data, transform)
    loss_value, _ = loss(data, data)
    assert 0 == pytest.approx(loss_value.numpy())
    transform.forward.assert_called()
