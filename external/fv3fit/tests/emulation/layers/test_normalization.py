from fv3fit.emulation.layers.normalization import (
    MeanMethod,
    NormLayer,
    StdDevMethod,
    _compute_scale,
    _compute_center,
)
import tensorflow as tf
import numpy as np
import pytest


def test_norm_layer():
    norm = NormLayer(scale=2.0, center=1.0)
    x = tf.ones([1])
    round_trip = norm.backward(norm.forward(x))
    np.testing.assert_almost_equal(round_trip, x)


def test_layers_no_trainable_variables():
    layer = NormLayer(1, 1)
    assert len(layer.trainable_variables) == 0


def test_fit_mean_per_feature():
    m = tf.ones([10, 4])
    mean = _compute_center(m, MeanMethod.per_feature)
    assert tuple(mean.shape) == (4,)


def test_fit_mean_none():
    m = tf.ones([10, 4])
    mean = _compute_center(m, MeanMethod.none)
    assert tuple(mean.shape) == ()


def test__compute_scale_all():
    m = tf.ones([10, 4])
    scale = _compute_scale(m, StdDevMethod.all)
    assert tuple(scale.shape) == ()


def test__compute_scale_max():
    m = tf.ones([10, 4])
    scale = _compute_scale(m, StdDevMethod.max)
    assert tuple(scale.shape) == ()


def test__compute_scale_per_feature():
    m = tf.ones([10, 4])
    scale = _compute_scale(m, StdDevMethod.per_feature)
    assert tuple(scale.shape) == (4,)


def test__compute_scale_none():
    m = tf.ones([10, 4])
    scale = _compute_scale(m, StdDevMethod.none)
    assert tuple(scale.shape) == ()


def _print_approx(arr, decimals=6, file=None):
    print((arr * 10 ** decimals).numpy().astype(int).tolist(), file=file)


@pytest.mark.parametrize("method", list(StdDevMethod))
def test__compute_scale_max_value(method: StdDevMethod, regtest):
    x = tf.cast([[1, -1], [1, -2], [3, -3]], tf.float32)
    scale = _compute_scale(x, method)
    with regtest:
        _print_approx(scale)


@pytest.mark.parametrize("method", list(MeanMethod))
def test__compute_center_values(method: MeanMethod, regtest):
    x = tf.cast([[1, -1], [1, -2], [3, -3]], tf.float32)
    center = _compute_center(x, method)
    with regtest:
        _print_approx(center)
