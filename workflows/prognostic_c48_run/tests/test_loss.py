from runtime.loss import MultiVariableLoss, ScalarLoss
import tensorflow as tf


def test_MultiVariableLoss():
    def model(x):
        return x

    in_ = [tf.ones((1, 10))] * 4
    out = [tf.ones((1, 10))] * 4

    loss, info = MultiVariableLoss().loss(model, in_, out)

    assert isinstance(loss, tf.Tensor)
    assert {
        "loss_u": 0.0,
        "loss_v": 0.0,
        "loss_q": 0.0,
        "loss_t": 0.0,
        "loss": 0.0,
    } == info


def test_ScalarLoss(regtest):
    varnum = 0
    i = 5

    def model(x):
        return x[varnum][:, i : i + 1]

    in_ = [tf.ones((1, 10))]
    out = in_

    loss, info = ScalarLoss(varnum, level=i).loss(model, in_, out)
    assert isinstance(loss, tf.Tensor)
    print(info, file=regtest)
