import sys
from dataclasses import asdict
from unittest.mock import Mock

import pytest
import tensorflow as tf
import yaml
from fv3fit._shared.config import OptimizerConfig, _to_flat_dict
from fv3fit.emulation import models
from fv3fit.emulation.data.config import TransformConfig
from fv3fit.emulation.layers.architecture import ArchitectureConfig
from fv3fit.emulation.losses import CustomLoss
from fv3fit.emulation.models import MicrophysicsConfig
from fv3fit.emulation.zhao_carr_fields import Field
from fv3fit.train_microphysics import TrainConfig, main
from fv3fit.wandb import WandBConfig


def _get_test_config():

    input_vars = [
        "air_temperature_input",
        "pressure_thickness_of_atmospheric_layer",
    ]

    output_vars = ["air_temperature_after_precpd"]

    model_config = models.MicrophysicsConfig(
        input_variables=input_vars, direct_out_variables=output_vars
    )

    transform = TransformConfig()

    loss = CustomLoss(
        optimizer=OptimizerConfig(name="Adam", kwargs=dict(learning_rate=1e-4)),
        loss_variables=output_vars,
        weights={output_vars[0]: 1.0},
    )

    config = TrainConfig(
        train_url="gs://vcm-ml-experiments/microphysics-emulation/2021-11-24/microphysics-training-data-v3-training_netcdfs/train",  # noqa E501
        test_url="gs://vcm-ml-experiments/microphysics-emulation/2021-11-24/microphysics-training-data-v3-training_netcdfs/test",  # noqa E501
        out_url="gs://vcm-ml-scratch/andrep/test-train-emulation",
        model=model_config,
        transform=transform,
        loss=loss,
        nfiles=80,
        nfiles_valid=80,
        valid_freq=1,
        epochs=4,
        wandb=WandBConfig(job_type="training"),
        input_variables=input_vars,
        data_variables=input_vars + loss.loss_variables,
    )

    return config


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

    config = _get_test_config()
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

    expected = _get_test_config()
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

    expected = _get_test_config()
    flat_dict = _to_flat_dict(asdict(expected))
    result = TrainConfig.from_flat_dict(flat_dict)
    assert result == expected


def test_TrainConfig_from_yaml(tmp_path):

    default = _get_test_config()

    yaml_path = str(tmp_path / "train_config.yaml")
    with open(yaml_path, "w") as f:
        yaml.safe_dump(asdict(default), f)

        loaded = TrainConfig.from_yaml_path(yaml_path)

        assert loaded == default


@pytest.fixture
def default_yml(tmpdir):
    config = _get_test_config()
    yml = tmpdir.join("default.yaml")
    with yml.open("w") as f:
        f.write(yaml.safe_dump(asdict(config)))
    return str(yml)


def test_TrainConfig_from_args_sysargv(monkeypatch, default_yml):

    args = [
        "unused_sysv_arg",
        "--config-path",
        default_yml,
        "--epochs",
        "4",
        "--model.architecture.name",
        "rnn",
    ]
    monkeypatch.setattr(sys, "argv", args)

    config = TrainConfig.from_args()

    assert config.epochs == 4
    assert config.model.architecture.name == "rnn"


@pytest.mark.parametrize(
    "arch_key, expected_cache",
    [("dense", True), ("rnn-v1", False), ("rnn-v1-shared-weights", False)],
)
def test_rnn_v1_cache_disable(arch_key, expected_cache):

    config = TrainConfig(
        ".",
        ".",
        ".",
        cache=True,
        model=MicrophysicsConfig(architecture=ArchitectureConfig(name=arch_key)),
    )
    assert config.cache == expected_cache


@pytest.mark.regression
def test_training_entry_integration(tmp_path):

    config = _get_test_config()
    config.out_url = str(tmp_path)
    config.use_wandb = False
    config.nfiles = 4
    config.nfiles_valid = 4
    config.epochs = 1

    main(config)


def test_TrainConfig_build_model():
    field = Field("out", "in")
    in_ = "in"
    out = "out"
    config = TrainConfig(
        ".",
        ".",
        ".",
        input_variables=[in_],
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
