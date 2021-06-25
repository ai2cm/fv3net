from runtime.emulator.loss import MultiVariableLoss, ScalarLoss
import tensorflow as tf
from .utils import _get_argsin


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


def test_ScalarLoss():
    varnum = 3
    i = 5

    def model(x):
        return x.q[:, i : i + 1]

    in_ = _get_argsin(levels=i + 1)
    out = in_
    loss, info = ScalarLoss(varnum, level=i).loss(model(in_), out)
    assert isinstance(loss, tf.Tensor)
    assert {
        "loss/variable_3/level_5": 0.0,
        "relative_humidity_mse/level_5": 0.0,
    } == info
