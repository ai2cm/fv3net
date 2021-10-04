import pytest
import numpy as np
import tensorflow as tf

from fv3fit.emulation.layers.architecture import (
    MLPBlock,
    RNNBlock,
    CombineInputs,
)


def _get_data(shape):

    num = int(np.prod(shape))
    return np.arange(num).reshape(shape).astype(np.float32)


def _get_tensor(shape):
    return tf.convert_to_tensor(_get_data(shape))


def test_MLPBlock():

    mlp = MLPBlock(width=256, depth=3)
    assert len(mlp.dense) == 3

    tensor = _get_tensor((20, 3))
    result = mlp(tensor)

    assert result.shape == (20, 256)


def test_MLPBlock_no_dense_layers():
    mlp = MLPBlock(width=256, depth=0)

    tensor = _get_tensor((20, 10))
    result = mlp(tensor)

    assert result.shape == (20, 10)


@pytest.mark.parametrize("depth,expected_shp", [(1, (20, 64)), (0, (20, 128))])
def test_RNNBlock(depth, expected_shp):

    rnn = RNNBlock(channels=128, dense_width=64, dense_depth=depth)

    tensor = _get_tensor((20, 10, 2))
    result = rnn(tensor)
    assert result.shape == expected_shp


def test_CombineInputs_no_expand():

    tensor = _get_tensor((20, 4))
    combiner = CombineInputs(-1, expand_axis=None)
    result = combiner((tensor, tensor))

    assert result.shape == (20, 8)
    np.testing.assert_array_equal(result[..., 4:8], tensor)


def test_CombineInputs_expand():

    tensor = _get_tensor((20, 4))
    combiner = CombineInputs(2, expand_axis=2)
    result = combiner((tensor, tensor, tensor))

    assert result.shape == (20, 4, 3)
    np.testing.assert_array_equal(result[..., 2], tensor)
