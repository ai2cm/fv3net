import pytest
import numpy as np
import tensorflow as tf

from fv3fit.emulation.layers.fields import (
    FieldInput,
    FieldOutput,
    IncrementedFieldOutput,
    IncrementStateLayer,
)


def _get_data(shape):

    num = int(np.prod(shape))
    return np.arange(num).reshape(shape).astype(np.float32)


def _get_tensor(shape):
    return tf.convert_to_tensor(_get_data(shape))


def test_FieldInput_no_args():

    tensor = _get_tensor((10, 5))
    field_in = FieldInput()
    result = field_in(tensor)

    np.testing.assert_array_equal(result, tensor)


def test_FieldInput():

    tensor = _get_tensor((10, 3))
    field_in = FieldInput(sample_in=tensor, normalize="mean_std", selection=slice(0, 2))

    result = field_in(tensor)
    assert result.shape == (10, 2)
    assert field_in.normalize.fitted
    assert np.max(abs(result)) < 2


def test_FieldOutput():

    net_tensor = _get_tensor((20, 64))
    sample = _get_tensor((20, 3))

    field_out = FieldOutput(sample.shape[-1], sample_out=sample, denormalize="mean_std")
    result = field_out(net_tensor)

    assert result.shape == (20, 3)
    assert field_out.denorm.fitted


def test_FieldOutput_no_norm():

    net_tensor = _get_tensor((20, 64))
    sample = _get_tensor((20, 3))

    field_out = FieldOutput(sample.shape[-1], sample_out=sample, denormalize=None)
    result = field_out(net_tensor)

    assert result.shape == (20, 3)


def test_increment_layer():

    in_ = tf.ones((2, 4), dtype=tf.float32)
    incr = tf.ones((2, 4), dtype=tf.float32)
    expected = tf.convert_to_tensor([[3] * 4, [3] * 4], dtype=tf.float32)

    incr_layer = IncrementStateLayer(2)
    incremented = incr_layer(in_, incr)

    assert incr_layer.dt_sec == 2
    np.testing.assert_array_equal(incremented, expected)


def test_IncrementedFieldOutput():

    net_tensor = _get_tensor((20, 64))
    sample = _get_tensor((20, 3))

    dt_sec = 2

    field_out = IncrementedFieldOutput(
        sample.shape[-1], dt_sec, sample_out=sample, denormalize="mean_std"
    )
    result = field_out(sample, net_tensor)
    tendency = field_out.get_tendency_output(net_tensor)

    assert result.shape == (20, 3)
    assert tendency.shape == (20, 3)
    assert field_out.tendency.denorm.fitted


def get_test_tensor():
    return _get_tensor((20, 10))


def get_FieldInput():

    tensor = get_test_tensor()
    input_layer = FieldInput(
        sample_in=tensor, normalize="mean_std", selection=slice(-3)
    )

    return input_layer


def get_FieldOutput():

    tensor = get_test_tensor()
    output_layer = FieldOutput(
        tensor.shape[-1],
        sample_out=tensor,
        denormalize="mean_std",
        enforce_positive=True,
    )

    return output_layer


def get_IncrementedStateOutput():

    tensor = get_test_tensor()
    layer = IncrementedFieldOutput(
        tensor.shape[-1],
        900,
        sample_out=tensor,
        denormalize="mean_std",
        enforce_positive=True,
    )
    return layer


@pytest.mark.parametrize("get_layer_func", [get_FieldInput, get_FieldOutput])
def test_layer_model_saving(tmpdir, get_layer_func):

    tensor = get_test_tensor()
    layer = get_layer_func()

    model = tf.keras.models.Sequential([layer, tf.keras.layers.Lambda(lambda x: x)])

    expected = model(tensor)
    model.save(tmpdir.join("model.tf"), save_format="tf")
    loaded = tf.keras.models.load_model(tmpdir.join("model.tf"))
    result = loaded(tensor)

    np.testing.assert_array_equal(result, expected)


def test_layer_IncrementedStateOutput_model_saving(tmpdir):

    tensor = get_test_tensor()

    in_ = tf.keras.layers.Input(tensor.shape[-1])
    dense = tf.keras.layers.Dense(64)(in_)
    out = get_IncrementedStateOutput()(in_, dense)
    model = tf.keras.models.Model(inputs=in_, outputs=out)

    expected = model(tensor)
    model.save(tmpdir.join("model.tf"), save_format="tf")
    loaded = tf.keras.models.load_model(tmpdir.join("model.tf"))
    result = loaded(tensor)

    np.testing.assert_array_equal(result, expected)
