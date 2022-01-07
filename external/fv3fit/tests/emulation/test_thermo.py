import pytest
import tensorflow as tf
from fv3fit.emulation.thermo import (
    conservative_precipitation_zhao_carr,
    relative_humidity,
    specific_humidity_from_rh,
)
from vcm.calc.constants import _RDGAS, _RVGAS


@pytest.mark.parametrize("celsius, rh", [(26, 0.5), (14.77, 1.0)])
def test_relative_humidity(celsius, rh):
    """
    Compare withh https://www.omnicalculator.com/physics/air-density
    """

    rho = 1.1781
    p = 1018_00
    e = 16_79.30

    q = _RDGAS / _RVGAS * e / (p - e)
    rho = tf.Variable(rho)

    T = tf.Variable(celsius + 273.15)
    ans = relative_humidity(T, q, rho)

    assert pytest.approx(rh, rel=0.03) == ans.numpy()


@pytest.mark.parametrize("t", [200, 250, 300])
@pytest.mark.parametrize("rh", [0, 0.5, 1.0])
@pytest.mark.parametrize("rho", [1.2, 1e-4])
def test_specific_humidity(t, rh, rho):
    """
    Compare withh https://www.omnicalculator.com/physics/air-density
    """
    q = specific_humidity_from_rh(t, rh, rho)
    rh_round_trip = relative_humidity(t, q, rho)

    assert pytest.approx(rh) == rh_round_trip


def test_conservative_precipitation():

    nz = 5
    expected_precip = 1.2

    one = tf.ones((1, nz))

    precip = conservative_precipitation_zhao_carr(
        one, one, one, one - expected_precip / nz, mass=1
    )
    assert precip.numpy() == pytest.approx(expected_precip / 1000)
