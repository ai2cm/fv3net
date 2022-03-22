from fv3fit.emulation.layers.normalization2 import (
    NormLayer,
    StdReduction,
    _fit_mean_per_feature,
    _compute_scale,
)
import tensorflow as tf
import numpy as np
import pytest


def test_norm_layer():
    norm = NormLayer(scale=2.0, center=1.0, epsilon=1.0)
    x = tf.ones([1], dtype=tf.float32)
    round_trip = norm.backward(norm.forward(x))
    np.testing.assert_almost_equal(round_trip, x)


def test_layers_no_trainable_variables():
    layer = NormLayer(1, 1)
    assert len(layer.trainable_variables) == 0


def test_fit_mean_per_feature():
    m = tf.ones([10, 4])
    assert tuple(_fit_mean_per_feature(m).shape) == (4,)


def test__compute_scale_all():
    m = tf.ones([10, 4])
    scale = _compute_scale(m, StdReduction.all)
    assert tuple(scale.shape) == ()


def test__compute_scale_max():
    m = tf.ones([10, 4])
    scale = _compute_scale(m, StdReduction.max)
    assert tuple(scale.shape) == ()


def test__compute_scale_per_feature():
    m = tf.ones([10, 4])
    scale = _compute_scale(m, StdReduction.per_feature)
    assert tuple(scale.shape) == (4,)


@pytest.mark.parametrize("method", list(StdReduction))
def test__compute_scale_max_value(method: StdReduction, regtest):
    x = tf.cast([[1, -1], [1, -2], [3, -3]], tf.float32)
    scale = _compute_scale(x, method)
    print(scale.numpy().tolist(), file=regtest)
