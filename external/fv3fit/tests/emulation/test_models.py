import tempfile
from os.path import join

import fv3fit.emulation.models
import numpy as np
import pytest
import tensorflow as tf
from fv3fit._shared import SliceConfig
from fv3fit.emulation import transforms
from fv3fit.emulation.layers import ArchitectureConfig
from fv3fit.emulation.layers.architecture import _ARCHITECTURE_KEYS
from fv3fit.emulation.models import MicrophysicsConfig, TransformedModelConfig
from fv3fit.emulation.zhao_carr_fields import Field


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
    m = {"dummy_out1": data, "dummy_in": data, "dummy_out1_tendency": data}
    model = config.build(m)
    output = model(data)
    assert set(output) == {"dummy_out1", "dummy_out1_tendency"}


def test_precip_conserving_config():
    factory = fv3fit.emulation.models.ConservativeWaterConfig()

    one = tf.ones((4, 5))

    data = {v: one for v in factory.input_variables + factory.output_variables}
    model = factory.build(data)
    out = model(data)
    precip = out[factory.fields.surface_precipitation.output_name]
    # scalar outputs need a singleton dimension to avoid broadcasting mayhem
    assert tuple(precip.shape) == (one.shape[0], 1)


def test_precip_conserving_output_variables():
    fields = fv3fit.emulation.models.ZhaoCarrFields(
        cloud_water=fv3fit.emulation.models.Field(input_name="a0", output_name="a"),
        specific_humidity=fv3fit.emulation.models.Field(
            input_name="a1", output_name="b"
        ),
        air_temperature=fv3fit.emulation.models.Field(input_name="a2", output_name="c"),
        surface_precipitation=fv3fit.emulation.models.Field(output_name="d"),
    )
    factory = fv3fit.emulation.models.ConservativeWaterConfig(fields=fields)

    assert set(factory.output_variables) == set("abcd")


def test_precip_conserving_extra_inputs():
    extra_names = "abcdef"
    extras = [fv3fit.emulation.models.Field(input_name=ch) for ch in extra_names]

    factory = fv3fit.emulation.models.ConservativeWaterConfig(
        extra_input_variables=extras
    )
    assert set(extra_names) < set(factory.input_variables)


@pytest.mark.parametrize("arch", _ARCHITECTURE_KEYS)
def test_MicrophysicConfig_model_save_reload(arch):

    config = MicrophysicsConfig(
        input_variables=["field_input"],
        direct_out_variables=["field_output"],
        architecture=ArchitectureConfig(name=arch),
    )

    nlev = 15
    data = tf.random.normal((10, nlev))
    sample = {"field_input": data, "field_output": data}

    model = config.build(sample)

    expected = model(sample)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = join(tmpdir, "model.tf")
        model.save(model_path, save_format="tf")
        reloaded = tf.keras.models.load_model(model_path, compile=False)

    result = reloaded(sample)
    np.testing.assert_array_equal(expected["field_output"], result["field_output"])


def test_RNN_downward_dependence():

    config = MicrophysicsConfig(
        input_variables=["field_input"],
        direct_out_variables=["field_output"],
        architecture=ArchitectureConfig(name="rnn-v1", kwargs=dict(channels=16)),
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


def test_transformed_model_without_transform():
    field = Field("a_out", "a")
    data = {field.input_name: tf.ones((1, 10)), field.output_name: tf.ones((1, 10))}
    config = TransformedModelConfig(ArchitectureConfig("dense"), [field], 900)
    model = config.build(data)
    out = model(data)
    assert set(out) == {field.output_name}


def test_transformed_model_with_transform():
    field = Field("a_out", "a")
    transformed_field = Field("transform_a_out", "transform_a")
    data = {field.input_name: tf.ones((1, 10)), field.output_name: tf.ones((1, 10))}
    model = TransformedModelConfig(
        ArchitectureConfig("dense"), [transformed_field], 900
    ).build(
        data,
        transform=transforms.PerVariableTransform(
            [
                transforms.TransformedVariableConfig(
                    field.input_name,
                    transformed_field.input_name,
                    transforms.LogTransform(),
                ),
                transforms.TransformedVariableConfig(
                    field.output_name,
                    transformed_field.output_name,
                    transforms.LogTransform(),
                ),
            ]
        ),
    )
    out = model(data)
    assert set(out) == {transformed_field.output_name, field.output_name}
