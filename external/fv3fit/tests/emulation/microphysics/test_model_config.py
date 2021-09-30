import pytest
import numpy as np
import tensorflow as tf

from fv3fit.emulation.microphysics import Config
from fv3fit.emulation.microphysics.models import (
    get_architecture_cls,
    ArchitectureConfig,
)


def _get_data(shape):

    num = int(np.prod(shape))
    return np.arange(num).reshape(shape).astype(np.float32)


def _get_tensor(shape):
    return tf.convert_to_tensor(_get_data(shape))


def test_get_architecture_unrecognized():

    with pytest.raises(KeyError):
        get_architecture_cls("not_an_arch")


def test_ArchParams():
    arch = ArchitectureConfig(name="dense", kwargs=dict(width=128))
    assert isinstance(arch.build, tf.keras.layers.Layer)


def test_ArchParams_bad_kwargs():
    arch = ArchitectureConfig(name="linear", kwargs=dict(not_a_kwarg="hi"))
    with pytest.raises(TypeError):
        arch.build


def test_Config():

    config = Config(input_variables=["dummy_in"], direct_out_variables=["dummy_out"])
    assert config.input_variables == ["dummy_in"]
    assert config.direct_out_variables == ["dummy_out"]


def test_Config_from_dict():
    config = Config.from_dict(
        dict(input_variables=["dummy_in"], direct_out_variables=["dummy_out"],)
    )
    assert config.input_variables == ["dummy_in"]
    assert config.direct_out_variables == ["dummy_out"]


def test_Config__processed_inputs():

    config = Config(
        input_variables=["dummy_in1", "dummy_in2"],
        direct_out_variables=["dummy_out"],
        selection_map={"dummy_in1": slice(0, 3)},
    )

    data = _get_data((20, 5))
    tensor = _get_tensor((20, 5))
    processed = config._get_processed_inputs((data, data), (tensor, tensor))

    assert len(processed) == 2
    assert processed[0].shape == (20, 3)
    assert processed[1].shape == data.shape


def test_Config__processed_inputs_normalized():

    config = Config(
        input_variables=["dummy_in1"], direct_out_variables=[], normalize_key="mean_std"
    )

    data = _get_data((20, 5))
    tensor = _get_tensor((20, 5))
    processed = config._get_processed_inputs((data,), (tensor,))

    # should be normalized
    assert not np.any(processed[0] > 2)


def test_Config__get_direct_outputs():

    net_out = _get_tensor((20, 64))
    data = _get_data((20, 5))

    config = Config(input_variables=["dummy_in"], direct_out_variables=["dummy_out1"],)

    outputs = config._get_direct_outputs((data,), net_out)

    assert len(outputs) == 1
    assert outputs[0].shape == (20, 5)


def test_Config__get_residual_outputs():

    net_out = _get_tensor((20, 64))
    data = _get_data((20, 5))

    config = Config(
        input_variables=["dummy_in"],
        residual_out_variables={"dummy_out1": "dummy_in"},
        tendency_outputs={"dummy_out1": "dummy_out1_tendency"},
    )

    res_map = {"dummy_out1": data}

    outputs = config._get_residual_outputs([data], net_out, res_map)
    assert len(outputs) == 2
    assert outputs[0].shape == (20, 5)
    assert outputs[1].shape == (20, 5)


def test_Config__get_outputs_denorm():

    net_out = _get_tensor((20, 64))
    data = _get_data((20, 5))

    config = Config(
        input_variables=[],
        direct_out_variables=["dummy_out1"],
        residual_out_variables={"dummy_out2": "dummy_in"},
        normalize_key="mean_std",
    )

    res_map = {"dummy_out2": data}

    (output,) = config._get_direct_outputs((data,), net_out)
    assert np.any(output > 2)

    (output,) = config._get_residual_outputs((data,), net_out, res_map)
    assert np.any(output > 2)


def test_Config_build():

    config = Config(input_variables=["dummy_in"], direct_out_variables=["dummy_out"],)

    data = _get_data((20, 5))
    model = config.build([data], sample_direct_out=[data])

    assert len(model.inputs) == 1
    assert len(model.outputs) == 1
    assert model.input_names[0] == "dummy_in"
    assert model.output_names[0] == "dummy_out"


def test_Config_build_residual_w_extra_tends_out():

    config = Config(
        input_variables=["dummy_in"],
        residual_out_variables={"dummy_out1": "dummy_in"},
        tendency_outputs={"dummy_out1": "dummy_out1_tendency"},
        architecture=ArchitectureConfig(name="dense"),
    )

    data = _get_data((20, 5))
    model = config.build([data], sample_residual_out=[data])

    assert len(model.inputs) == 1
    assert len(model.outputs) == 2
    assert model.input_names[0] == "dummy_in"
    assert model.output_names[0] == "dummy_out1"
    assert model.output_names[1] == "dummy_out1_tendency"
