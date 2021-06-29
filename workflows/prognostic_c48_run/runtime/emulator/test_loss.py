from workflows.prognostic_c48_run.runtime.emulator.loss import RHLoss
from runtime.emulator.loss import MultiVariableLoss, ScalarLoss
import tensorflow as tf
from .utils import _get_argsin
import pytest


def test_MultiVariableLoss():

    tf.random.set_seed(1)
    in_ = _get_argsin(levels=10)
    loss, info = MultiVariableLoss().loss(in_, in_)

    assert isinstance(loss, tf.Tensor)
    assert {
        "loss_u": 0.0,
        "loss_v": 0.0,
        "loss_q": 0.0,
        "loss_t": 0.0,
        "loss": 0.0,
    } == info


@pytest.mark.parametrize("loss_fn", [ScalarLoss(3, 5), RHLoss(5)])
def test_ScalarLoss(loss_fn):
    i = loss_fn.level

    def model(x):
        if isinstance(loss_fn, ScalarLoss):
            return x.q[:, i : i + 1]
        elif isinstance(loss_fn, RHLoss):
            return x.rh[:, i : i + 1]

    in_ = _get_argsin(levels=i + 1)
    out = in_
    loss, info = loss_fn.loss(model(in_), out)
    assert isinstance(loss, tf.Tensor)
    assert f"loss/variable_3/level_{i}" in info
    assert f"relative_humidity_mse/level_{i}" in info
