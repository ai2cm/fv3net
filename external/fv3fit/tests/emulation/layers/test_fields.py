import numpy as np
import tensorflow as tf

from fv3fit.emulation.layers.fields import (
    CombineInputs,
    FieldInput,
    FieldOutput,
    IncrementedFieldOutput,
    IncrementStateLayer,
)


def _get_tensor(nsamples, nfeatures):
    return tf.convert_to_tensor(
        np.arange(nsamples * nfeatures).reshape(nfeatures, nsamples).T, dtype=tf.float32
    )


def test_FieldInput_no_args():

    tensor = _get_tensor(10, 5)
    field_in = FieldInput()
    result = field_in(tensor)

    np.testing.assert_array_equal(result, tensor)


def test_FieldInput():

    tensor = _get_tensor(10, 3)
    field_in = FieldInput(sample_in=tensor, normalize="mean_std", selection=slice(0, 2))

    result = field_in(tensor)
    assert result.shape == (10, 2)
    assert field_in.normalize.fitted
    assert np.max(abs(result)) < 2


def test_FieldOutput():

    net_tensor = _get_tensor(20, 64)
    sample = _get_tensor(20, 3)

    field_out = FieldOutput(sample, denormalize="mean_std")
    result = field_out(net_tensor)

    assert result.shape == (20, 3)
    assert field_out.denorm.fitted


def test_FieldOutput_no_norm():

    net_tensor = _get_tensor(20, 64)
    sample = _get_tensor(20, 3)

    field_out = FieldOutput(sample, denormalize=None)
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

    net_tensor = _get_tensor(20, 64)
    sample = _get_tensor(20, 3)

    dt_sec = 2

    field_out = IncrementedFieldOutput(sample, dt_sec, denormalize="mean_std")
    result = field_out(sample, net_tensor)
    tendency = field_out.get_tendency_output(net_tensor)

    assert result.shape == (20, 3)
    assert tendency.shape == (20, 3)
    assert field_out.tendency.denorm.fitted


def test_CombineInputs_no_expand():

    tensor = _get_tensor(20, 4)
    combiner = CombineInputs(-1, expand_axis=None)
    result = combiner((tensor, tensor))

    assert result.shape == (20, 8)
    np.testing.assert_array_equal(result[..., 4:8], tensor)


def test_CombineInputs_expand():

    tensor = _get_tensor(20, 4)
    combiner = CombineInputs(2, expand_axis=2)
    result = combiner((tensor, tensor, tensor))

    assert result.shape == (20, 4, 3)
    np.testing.assert_array_equal(result[..., 2], tensor)
