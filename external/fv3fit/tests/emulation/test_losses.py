import pytest
import numpy as np
import tensorflow as tf
from fv3fit.emulation.losses import (
    CustomLoss,
    NormalizedMSE,
)


def test_NormalizeMSE():
    sample = np.array([[25.0], [75.0]])
    target = np.array([[50.0], [50.0]])

    mse_func = NormalizedMSE("mean_std", sample)
    mse = mse_func(target, sample)
    np.testing.assert_approx_equal(mse, 1.0, 6)


def test_CustomLoss():
    loss_fn = CustomLoss(
        normalization="mean_std",
        loss_variables=["fieldA", "fieldB"],
        metric_variables=["fieldC"],
        weights=dict(fieldA=2.0),
    )

    tensor = tf.random.normal((100, 2))

    names = ["fieldA", "fieldB", "fieldC", "fieldD"]
    samples = [tensor] * 4
    m = dict(zip(names, samples))
    loss_fn.prepare(m)

    # make a copy with some error
    compare = m.copy()

    loss, info = loss_fn(m, compare)
    all_loss_vars = set(loss_fn.loss_variables) | set(loss_fn.metric_variables)
    expected_variables = set(v + "_loss" for v in all_loss_vars)
    assert set(info) == expected_variables
    assert loss.numpy() == pytest.approx(0.0)
