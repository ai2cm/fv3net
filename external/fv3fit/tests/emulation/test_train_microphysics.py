import pathlib
import pytest
import tensorflow as tf
from pathlib import PosixPath
from unittest.mock import Mock

import fv3fit.emulation.transforms.zhao_carr as zhao_carr
from fv3fit.dataclasses import asdict_with_enum as asdict
from fv3fit._shared.training_config import to_flat_dict
from fv3fit.emulation.data.config import TransformConfig
from fv3fit.emulation.layers.architecture import ArchitectureConfig
from fv3fit.emulation.losses import CustomLoss
from fv3fit.emulation.models import MicrophysicsConfig
from fv3fit.emulation.zhao_carr_fields import Field
from fv3fit.emulation.zhao_carr.filters import HighAntarctic
from fv3fit.emulation.transforms import GscondRoute
import fv3fit.train_microphysics as api


def get_default_config():

    input_vars = [
        "air_temperature_input",
        "specific_humidity_input",
        "cloud_water_mixing_ratio_input",
        "pressure_thickness_of_atmospheric_layer",
    ]
    tensor_transforms = [
        api.Difference(
            to="temperature_diff",
            before="air_temperature_input",
            after="air_temperature_after_precpd",
        ),
        api.Difference(
            to="humidity_diff",
            before="specific_humidity_input",
            after="specific_humidity_after_precpd",
        ),
    ]

    model_config = MicrophysicsConfig(
        input_variables=input_vars,
        direct_out_variables=[
            "cloud_water_mixing_ratio_after_precpd",
            "total_precipitation",
            "temperature_diff",
            "humidity_diff",
        ],
        architecture=ArchitectureConfig("linear"),
        selection_map=dict(
            air_temperature_input=api.SliceConfig(stop=-10),
            specific_humidity_input=api.SliceConfig(stop=-10),
            cloud_water_mixing_ratio_input=api.SliceConfig(stop=-10),
            pressure_thickness_of_atmospheric_layer=api.SliceConfig(stop=-10),
        ),
    )

    transform = TransformConfig()

    loss = CustomLoss(
        optimizer=api.OptimizerConfig(name="Adam", kwargs=dict(learning_rate=1e-4)),
        loss_variables=[
            "air_temperature_after_precpd",
            "specific_humidity_after_precpd",
            "cloud_water_mixing_ratio_after_precpd",
            "total_precipitation",
        ],
        weights=dict(
            air_temperature_after_precpd=0.5e5,
            specific_humidity_after_precpd=0.5e5,
            cloud_water_mixing_ratio_after_precpd=1.0,
            total_precipitation=0.04,
        ),
        metric_variables=["temperature_diff"],
    )

    config = api.TrainConfig(
        train_url="gs://vcm-ml-code-testing-data/microphysics/train",  # noqa E501
        test_url="gs://vcm-ml-code-testing-data/microphysics/train",  # noqa E501
        out_url="gs://vcm-ml-scratch/andrep/test-train-emulation",
        model=model_config,
        tensor_transform=tensor_transforms,
        transform=transform,
        loss=loss,
        nfiles=30,
        nfiles_valid=30,
        valid_freq=1,
        epochs=4,
        wandb=api.WandBConfig(job_type="training"),
    )

    return config


def test_TrainConfig_defaults():

    config = api.TrainConfig(
        train_url="train_path",
        test_url="test_path",
        out_url="save_path",
        transform=TransformConfig(),
        model=MicrophysicsConfig(),
    )

    assert config  # for linter


def test_get_default_config():

    config = get_default_config()
    assert isinstance(config, api.TrainConfig)


def test_TrainConfig_asdict():

    config = api.TrainConfig(
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

    config = api.TrainConfig.from_dict(d)
    assert config.train_url == "train_path"
    assert config.model.architecture.name == "rnn"


def test_TrainConfig_from_dict_full():

    expected = get_default_config()
    result = api.TrainConfig.from_dict(asdict(expected))

    assert result == expected


def test_TrainConfig_from_flat_dict():

    d = {
        "train_url": "train_path",
        "test_url": "test_path",
        "out_url": "out_path",
        "model.architecture.name": "rnn",
    }

    config = api.TrainConfig.from_flat_dict(d)

    assert config.train_url == "train_path"
    assert config.model.architecture.name == "rnn"

    expected = get_default_config()
    flat_dict = to_flat_dict(asdict(expected))
    result = api.TrainConfig.from_flat_dict(flat_dict)
    assert result == expected


def test_TrainConfig_from_yaml(tmp_path: PosixPath):
    default = api.TrainConfig(test_url=".", train_url=".", out_url=".")
    yaml_path = tmp_path / "train_config.yaml"
    yaml_path.write_text(default.to_yaml())
    loaded = api.TrainConfig.from_yaml_path(yaml_path.as_posix())
    assert loaded == default


def test_TrainConfig_from_args(tmp_path: pathlib.Path):

    config = get_default_config()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config.to_yaml())

    args = [
        "--config-path",
        config_path.as_posix(),
        "--epochs",
        "4",
        "--model.architecture.name",
        "rnn",
    ]
    config = api.TrainConfig.from_args(args)

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

    config = api.TrainConfig.from_dict(config_dict)

    api.main(config)


def test_MicrophysicsHyperParameters_build_model():
    field = Field("out", "in")
    in_ = "in"
    out = "out"
    config = api.TransformedParameters(
        model=api.MicrophysicsConfig(
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
    config = api.TrainConfig(".", ".", ".", loss=CustomLoss(loss_variables=["x"]))
    # needs to be random or the normalized loss will have nan
    data = {"x": tf.random.uniform(shape=(4, 10))}
    transform = Mock()
    transform.forward.return_value = data
    loss = config.build_loss(data, transform)
    loss_value, _ = loss(data, data)
    assert 0 == pytest.approx(loss_value.numpy())
    transform.forward.assert_called()


def test_TrainConfig_GscondClassesV1():
    timestep = 1.0
    config_dict = {"tensor_transform": [{"timestep": timestep}]}
    config = api.TrainConfig.from_dict(config_dict)

    data = {
        "cloud_water_mixing_ratio_input": tf.ones((1, 4)),
        "cloud_water_mixing_ratio_after_gscond": tf.ones((1, 4),),
    }
    transform = config.build_transform(sample=data)
    result = transform.forward(data)

    # Check the length so it errors if classes added/removed without top-level
    # definitions being updated
    assert set(result) >= set(zhao_carr.CLASS_NAMES) | {zhao_carr.NONTRIVIAL_TENDENCY}


def test_TrainConfig_model_variables_with_backwards_transform():
    route = GscondRoute()
    config = api.TrainConfig(tensor_transform=[route], model=api.MicrophysicsConfig())
    assert config.model_variables >= route.backward_input_names()


def test_TrainConfig_input_variables_with_backwards_transform():
    route = GscondRoute()
    config = api.TrainConfig(tensor_transform=[route], model=api.MicrophysicsConfig())
    assert set(config.input_variables) >= route.backward_input_names()


def test_TrainConfig_inputs_routed():
    this_file = PosixPath(__file__)
    path = this_file.parent / "gscond-only-routed.yaml"
    config = api.TrainConfig.from_yaml_path(path.as_posix())
    assert "air_temperature_after_gscond" not in config.input_variables
    assert "specific_humidity_after_gscond" not in config.input_variables
    assert "cloud_water_mixing_ratio_after_gscond" not in config.input_variables


def test_TrainConfig_filters():
    config = api.TrainConfig(filters=[HighAntarctic()], model=api.MicrophysicsConfig())
    assert {"latitude", "surface_air_pressure"} <= config.model_variables
