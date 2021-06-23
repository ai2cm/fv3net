from runtime.emulator.loss import MultiVariableLoss, ScalarLoss
import tensorflow as tf


def test_MultiVariableLoss():
    def model(x):
        return x[:4]

    in_ = [tf.ones((1, 10))] * 5
    out = [tf.ones((1, 10))] * 5

    loss, info = MultiVariableLoss().loss(model, in_, out)

    assert isinstance(loss, tf.Tensor)
    assert {
        "loss_u": 0.0,
        "loss_v": 0.0,
        "loss_q": 0.0,
        "loss_t": 0.0,
        "loss": 0.0,
    } == info


def test_ScalarLoss():
    varnum = 0
    i = 5

    def model(x):
        return x[varnum][:, i : i + 1]

    in_ = [tf.ones((1, 10))] * 5
    out = in_

    loss, info = ScalarLoss(varnum, level=i).loss(model, in_, out)
    assert isinstance(loss, tf.Tensor)
    assert {"loss/variable_0/level_5": 0.0, "relative_humidity_mse": 0.0} == info
