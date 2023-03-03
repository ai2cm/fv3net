import pytest
import numpy as np
import tensorflow as tf

from fv3fit.emulation.layers.fields import (
    FieldInput,
    FieldOutput,
)
from fv3fit.emulation.layers.normalization2 import norm2_factory_from_key


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
    field_in = FieldInput(
        sample_in=tensor,
        normalize=norm2_factory_from_key("mean_std"),
        selection=slice(0, 2),
    )

    result = field_in(tensor)
    assert result.shape == (10, 2)
    assert np.max(abs(result)) < 2


def test_FieldOutput():
    sample = _get_tensor((20, 3))

    field_out = FieldOutput(
        sample_out=sample, denormalize=norm2_factory_from_key("mean_std")
    )
    result = field_out(sample)

    assert result.shape == (20, 3)
    assert tf.math.reduce_std(sample) < tf.math.reduce_std(result)


def test_FieldOutput_no_norm():

    sample = _get_tensor((20, 3))

    field_out = FieldOutput(sample_out=sample, denormalize=None)
    result = field_out(sample)

    assert result.shape == (20, 3)
    np.testing.assert_array_equal(sample, result)


def get_test_tensor():
    return _get_tensor((20, 10))


def get_FieldInput():

    tensor = get_test_tensor()
    input_layer = FieldInput(
        sample_in=tensor,
        normalize=norm2_factory_from_key("mean_std"),
        selection=slice(-3),
    )

    return input_layer


def get_FieldOutput():

    tensor = get_test_tensor()
    output_layer = FieldOutput(
        sample_out=tensor, denormalize=norm2_factory_from_key("mean_std")
    )

    return output_layer


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
