import pytest
import numpy as np
import tensorflow as tf
from fv3fit.emulation.losses import CustomLoss, NormalizedMSE, bias
from fv3fit.emulation.layers.normalization import (
    NormFactory,
    StdDevMethod,
    MeanMethod,
    NormLayer,
)


def test_NormalizeMSE():
    sample = np.array([[0], [0]])
    target = np.array([[1], [2]])
    mse_func = NormalizedMSE(NormLayer(1, 0))
    assert tf.is_tensor(mse_func(target, sample))


def test_CustomLoss():
    loss_config = CustomLoss(
        normalization=NormFactory(StdDevMethod.all, MeanMethod.per_feature),
        loss_variables=["fieldA", "fieldB"],
        metric_variables=["fieldC"],
        weights=dict(fieldA=2.0),
    )

    tensor = tf.random.normal((100, 2))

    names = ["fieldA", "fieldB", "fieldC", "fieldD"]
    samples = [tensor] * 4
    m = dict(zip(names, samples))
    loss_fn = loss_config.build(m)

    # make a copy with some error
    compare = m.copy()

    loss, info = loss_fn(m, compare)
    all_loss_vars = set(loss_fn.loss_variables) | set(loss_fn.metric_variables)
    expected_variables = set(v + "_loss" for v in all_loss_vars)
    assert set(info) == expected_variables
    assert loss.numpy() == pytest.approx(0.0)


def test_keras_categorical_cross_entropy():
    x = tf.ones([4, 5, 3])
    y = tf.random.normal(shape=[4, 5, 3])
    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    loss = cce(x, y)
    assert tuple(loss.shape) == ()


def test_custom_loss_logits():
    config = CustomLoss(logit_variables=["a"])
    loss = config.build({})
    assert "a" in loss.loss_funcs
    assert "a" in loss.loss_variables


def test_bias():
    x = tf.ones([2, 2])
    y = tf.zeros([2, 2])
    result = bias(x, y)
    assert tuple(result.shape) == ()
    assert result.numpy() == pytest.approx(-1.0)


def test_custom_loss_bias():
    config = CustomLoss(bias_metric_variables=["a"])
    loss = config.build({})
    assert "a" in loss.bias_variables

    true = {"a": tf.ones([2, 2])}
    pred = {"a": tf.ones([2, 2])}
    loss_values, metric_values = loss(true, pred)
    assert not loss_values
    assert "a_bias" in metric_values
