import tensorflow as tf
from runtime.thermo import relative_humidity
from vcm.calc.thermo import _GRAVITY, _RDGAS, _RVGAS
import pytest


@pytest.mark.parametrize("celsius, rh", [(26, 0.5), (14.77, 1.0)])
def test_relative_humidity(celsius, rh):
    """
    Compare withh https://www.omnicalculator.com/physics/air-density
    """

    rho = 1.1781
    p = 1018_00
    e = 16_79.30

    q = _RDGAS / _RVGAS * e / (p - e)
    delp = tf.Variable(rho * _GRAVITY)

    T = tf.Variable(celsius + 273.15)
    ans = relative_humidity(T, q, delp)

    assert pytest.approx(rh, rel=0.03) == ans.numpy()
