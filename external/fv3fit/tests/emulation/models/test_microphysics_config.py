import numpy as np
import pytest
import tensorflow as tf

from fv3fit._shared import SliceConfig
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


def test_Config_build():

    config = MicrophysicsConfig(
        input_variables=["dummy_in"], direct_out_variables=["dummy_out"],
    )

    data = _get_data((20, 5))
    m = {"dummy_in": data, "dummy_out": data}
    model = config.build(m)
    output = model(m)
    assert set(output) == {"dummy_out"}


def test_Config_build_residual_w_extra_tends_out():

    config = MicrophysicsConfig(
        input_variables=["dummy_in"],
        residual_out_variables={"dummy_out1": "dummy_in"},
        tendency_outputs={"dummy_out1": "dummy_out1_tendency"},
        architecture=ArchitectureConfig(name="dense"),
    )

    data = _get_data((20, 5))
    m = {"dummy_out1": data, "dummy_in": data}
    model = config.build(m)
    output = model(data)
    assert set(output) == {"dummy_out1", "dummy_out1_tendency"}


@pytest.mark.xfail
def test_RNN_downward_dependence():

    config = MicrophysicsConfig(
        input_variables=["field_input"],
        direct_out_variables=["field_output"],
        architecture=ArchitectureConfig(
            name="rnn",
            kwargs=dict(channels=16, dense_width=16, dense_depth=1)
        )
    )

    nlev = 15
    data = tf.random.normal((10, nlev))
    sample = {"field_input": data, "field_output": data}
    profile = data[0:1]

    model = config.build(sample)

    with tf.GradientTape() as g:
        g.watch(profile)
        output = model(profile)

    jacobian = g.jacobian(output["field_output"], profile)[0, :, 0]

    assert jacobian.shape == (nlev, nlev)
    for output_level in range(nlev):
        for input_level in range(nlev):
            sensitivity = jacobian[output_level, input_level]
            if output_level > input_level and sensitivity != 0:
                raise ValueError("Downwards dependence violated")
