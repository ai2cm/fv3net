import pytest
import numpy as np
import tensorflow as tf

from fv3fit.emulation.microphysics.layers import (
    CombineInputs,
    FieldInput,
    FieldOutput,
    MLPBlock,
    RNNBlock,
    IncrementedFieldOutput,
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


def test_ResidualOutput():

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


def test_MLPBlock():

    mlp = MLPBlock(width=256, depth=3)
    assert len(mlp.dense) == 3

    tensor = _get_tensor(20, 3)
    result = mlp(tensor)

    assert result.shape == (20, 256)


def test_MLPBlock_no_dense_layers():
    mlp = MLPBlock(width=256, depth=0)

    tensor = _get_tensor(20, 10)
    result = mlp(tensor)

    assert result.shape == (20, 10)


@pytest.mark.parametrize("depth,expected_shp", [(1, (20, 64)), (0, (20, 128))])
def test_RNNBlock(depth, expected_shp):

    rnn = RNNBlock(channels=128, dense_width=64, dense_depth=depth)

    tensor = _get_tensor(20, 10)
    combined = CombineInputs(-1, expand_axis=-1)((tensor, tensor))
    result = rnn(combined)
    assert result.shape == expected_shp
