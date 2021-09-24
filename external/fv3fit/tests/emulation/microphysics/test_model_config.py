import pytest
import numpy as np
import tensorflow as tf

from fv3fit.emulation.microphysics import Config
from fv3fit.emulation.microphysics.models import get_architecture_cls


def test_get_architecture_unrecognized():

    with pytest.raises(KeyError):
        get_architecture_cls("not_an_arch")


def test_Config():

    config = Config(input_variables=["dummy_in"], output_variables=["dummy_out"])
    assert config.input_variables == ["dummy_in"]
    assert config.output_variables == ["dummy_out"]


def test_Config_from_dict():
    config = Config.from_dict(
        dict(input_variables=["dummy_in"], output_variables=["dummy_out"],)
    )
    assert config.input_variables == ["dummy_in"]
    assert config.output_variables == ["dummy_out"]


def test_Config__processed_inputs():

    config = Config(
        input_variables=["dummy_in1", "dummy_in2"],
        output_variables=["dummy_out"],
        selection_map={"dummy_in1": slice(0, 3)},
    )

    data = np.ones((20, 5))
    tensor = tf.convert_to_tensor(data)
    processed = config._get_processed_inputs((data, data), (tensor, tensor))

    assert len(processed) == 2
    assert processed[0].shape == (20, 3)
    assert processed[1].shape == (20, 5)


def test_Config__get_outputs():

    net_out = tf.convert_to_tensor(np.ones((20, 64)))
    data = np.ones((20, 5))

    config = Config(
        input_variables=["dummy_in"],
        output_variables=["dummy_out1", "dummy_out2"],
        timestep_increment_sec=5,
    )

    residual_map = {"dummy_out2": data[..., :3]}

    outputs = config._get_outputs((data, data[..., :3]), net_out, residual_map)

    assert len(outputs) == 2
    assert outputs[0].shape == (20, 5)
    assert outputs[1].shape == (20, 3)


def test_Config__get_outputs_with_tendencies():

    net_out = tf.convert_to_tensor(np.ones((20, 64)))
    data = np.ones((20, 5))

    config = Config(
        input_variables=["dummy_in"],
        output_variables=["dummy_out1", "dummy_out2"],
        timestep_increment_sec=5,
        tendency_outputs={"dummy_out2": "tendency_of_dummy_due_to_X"},
    )

    residual_map = {"dummy_out2": data[..., :3]}

    outputs = config._get_outputs((data, data[..., :3]), net_out, residual_map)

    assert len(outputs) == 3
    assert outputs[0].shape == (20, 5)
    assert outputs[1].shape == (20, 3)
    assert outputs[2].shape == (20, 3)


def test_Config_build():

    config = Config(input_variables=["dummy_in"], output_variables=["dummy_out"],)

    data = np.random.randn(20, 5)
    model = config.build([data], [data])

    assert len(model.inputs) == 1
    assert len(model.outputs) == 1
    assert model.input_names[0] == "dummy_in"
    assert model.output_names[0] == "dummy_out"


def test_Config_build_residual_w_extra_tends_out():

    config = Config(
        input_variables=["dummy_in"],
        output_variables=["dummy_out"],
        architecture="dense",
        residual_to_state={"dummy_out": "dummy_in"},
        tendency_outputs={"dummy_out": "tendency_of_dummy_out"},
    )

    data = np.random.randn(20, 5)
    model = config.build([data], [data])

    assert len(model.inputs) == 1
    assert len(model.outputs) == 2
    assert model.input_names[0] == "dummy_in"
    assert model.output_names[0] == "dummy_out"
    assert model.output_names[1] == "tendency_of_dummy_out"
