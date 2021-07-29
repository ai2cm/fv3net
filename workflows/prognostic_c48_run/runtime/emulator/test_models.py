from runtime.emulator.thermo import RelativeHumidityBasis
import tensorflow as tf
from runtime.emulator.models import get_model, V1QCModel
from .utils import _get_argsin

import pytest


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


@pytest.mark.parametrize("with_scalars", [True, False])
def test_V1QCModel(with_scalars):
    x = _get_argsin(3)

    if with_scalars:
        n_scalars = 2
        x.scalars = [tf.random.uniform([x.q.shape[0], n_scalars])]
    else:
        n_scalars = 0

    model = V1QCModel(x.u.shape[1], n_scalars)
    assert not model.scalers_fitted
    model.fit_scalers(x, x)
    assert model.scalers_fitted
    y = model(x)
    assert isinstance(y, RelativeHumidityBasis)
