import pytest
import numpy as np
import tensorflow as tf
from fv3fit.emulation.losses import (
    CustomLoss,
    NormalizedMSE,
)
from fv3fit.emulation.layers.normalization2 import (
    NormFactory,
    ScaleMethod,
    CenterMethod,
    NormLayer,
)


def test_NormalizeMSE():
    sample = np.array([[0], [0]])
    target = np.array([[1], [2]])
    mse_func = NormalizedMSE(NormLayer(1, 0))
    assert tf.is_tensor(mse_func(target, sample))


def test_CustomLoss():
    loss_config = CustomLoss(
        normalization=NormFactory(ScaleMethod.all, CenterMethod.per_feature),
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
