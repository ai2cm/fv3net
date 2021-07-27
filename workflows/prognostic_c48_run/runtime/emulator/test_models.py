from runtime.emulator.thermo import RelativeHumidityBasis
import tensorflow as tf
from runtime.emulator.models import get_model, V1QCModel
from .utils import _get_argsin


def test_get_model():
    nz = 10
    num_scalar = 5

    u = tf.random.normal([2, nz])
    v = tf.random.normal([2, nz])
    t = tf.random.normal([2, nz])
    q = tf.random.normal([2, nz])
    qc = tf.random.normal([2, nz])
    scalars = tf.random.normal([2, num_scalar])

    model = get_model(nz, num_scalar)
    y = model([u, v, t, q, qc, scalars])
    assert tf.is_tensor(y)
    assert tuple(y.shape) == tuple(qc.shape)


def test_V1QCModel():
    x = _get_argsin(10)
    model = V1QCModel(10, 0)
    assert not model.scalers_fitted
    model.fit_scalers(x, x)
    assert model.scalers_fitted
    y = model(x)
    assert isinstance(y, RelativeHumidityBasis)
