import numpy as np
from fv3fit.emulation.thermobasis.loss import QVLoss, RHLoss, MultiVariableLoss
import tensorflow as tf
from utils import _get_argsin
import pytest


def test_MultiVariableLoss(regtest):

    tf.random.set_seed(1)
    in_ = _get_argsin(levels=10)
    loss, info = MultiVariableLoss(levels=[5, 9]).loss(in_, in_)

    assert isinstance(loss, tf.Tensor)
    print(info, file=regtest)

    for key in sorted(info):
        value = info[key]
        assert not np.isnan(value), key
        print(key, value, file=regtest)


@pytest.mark.parametrize("loss_fn", [QVLoss(5), RHLoss(5)])
def test_ScalarLoss(loss_fn):
    i = loss_fn.level

    def model(x):
        if isinstance(loss_fn, QVLoss):
            return x.q[:, i : i + 1]
        elif isinstance(loss_fn, RHLoss):
            return x.rh[:, i : i + 1]

    in_ = _get_argsin(levels=i + 1)
    out = in_
    loss, info = loss_fn.loss(model(in_), out)
    assert isinstance(loss, tf.Tensor)
    assert f"loss/variable_3/level_{i}" in info
    assert f"relative_humidity_mse/level_{i}" in info
