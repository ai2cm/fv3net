from fv3fit.emulation.data.config import SliceConfig
import numpy as np
import tensorflow as tf

from fv3fit.emulation.models import MicrophysicsConfig
from fv3fit.emulation.models._core import ArchitectureConfig


def _get_data(shape):

    num = int(np.prod(shape))
    return np.arange(num).reshape(shape).astype(np.float32)


def _get_tensor(shape):
    return tf.convert_to_tensor(_get_data(shape))


def test_Config():

    config = MicrophysicsConfig(
        input_variables=["dummy_in"], direct_out_variables=["dummy_out"]
    )
    assert config.input_variables == ["dummy_in"]
    assert config.direct_out_variables == ["dummy_out"]


def test_Config_from_dict():
    config = MicrophysicsConfig.from_dict(
        dict(input_variables=["dummy_in"], direct_out_variables=["dummy_out"],)
    )
    assert config.input_variables == ["dummy_in"]
    assert config.direct_out_variables == ["dummy_out"]


def test_Config_from_dict_selection_map_sequences():
    config = MicrophysicsConfig.from_dict(
        dict(selection_map=dict(dummy=dict(start=0, stop=2, step=1)))
    )
    assert config.selection_map["dummy"].slice == slice(0, 2, 1)


def test_Config_asdict():
    sl1_kwargs = dict(start=0, stop=10, step=2)
    sl2_kwargs = dict(start=None, stop=25, step=None)
    sel_map = dict(
        dummy_in=SliceConfig(**sl1_kwargs), dummy_out=SliceConfig(**sl2_kwargs)
    )

    original = MicrophysicsConfig(
        input_variables=["dummy_in"],
        direct_out_variables=["dummy_out"],
        selection_map=sel_map,
    )

    config_d = original.asdict()
    assert config_d["selection_map"]["dummy_in"] == sl1_kwargs
    assert config_d["selection_map"]["dummy_out"] == sl2_kwargs

    result = MicrophysicsConfig.from_dict(config_d)
    assert result == original


def test_Config__processed_inputs():

    config = MicrophysicsConfig(
        input_variables=["dummy_in1", "dummy_in2"],
        direct_out_variables=["dummy_out"],
        selection_map={"dummy_in1": SliceConfig(start=0, stop=3)},
    )

    data = _get_data((20, 5))
    tensor = _get_tensor((20, 5))
    processed = config._get_processed_inputs((data, data), (tensor, tensor))

    assert len(processed) == 2
    assert processed[0].shape == (20, 3)
    assert processed[1].shape == data.shape


def test_Config__processed_inputs_normalized():

    config = MicrophysicsConfig(
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

    config = MicrophysicsConfig(
        input_variables=["dummy_in"], direct_out_variables=["dummy_out1"],
    )

    outputs = config._get_direct_outputs((data,), net_out)

    assert len(outputs) == 1
    assert outputs[0].shape == (20, 5)


def test_Config__get_residual_outputs():

    net_out = _get_tensor((20, 64))
    data = _get_data((20, 5))

    config = MicrophysicsConfig(
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

    config = MicrophysicsConfig(
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

    config = MicrophysicsConfig(
        input_variables=["dummy_in"], direct_out_variables=["dummy_out"],
    )

    data = _get_data((20, 5))
    model = config.build([data], sample_direct_out=[data])

    assert len(model.inputs) == 1
    assert len(model.outputs) == 1
    assert model.input_names[0] == "dummy_in"
    assert model.output_names[0] == "dummy_out"


def test_Config_build_residual_w_extra_tends_out():

    config = MicrophysicsConfig(
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
