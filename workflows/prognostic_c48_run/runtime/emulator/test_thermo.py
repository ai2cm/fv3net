import tensorflow as tf
from runtime.emulator.thermo import (
    SpecificHumidityBasis,
    relative_humidity,
    specific_humidity_from_rh,
)
from vcm.calc.thermo import _RDGAS, _RVGAS
import pytest

from hypothesis import given
from hypothesis.strategies import floats, integers, sampled_from


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


@given(floats(200, 400), floats(0, 1), floats(1, 100))
def test_specific_humidity(t, rh, rho):
    """
    Compare withh https://www.omnicalculator.com/physics/air-density
    """
    q = specific_humidity_from_rh(t, rh, rho)
    rh_round_trip = relative_humidity(t, q, rho)

    assert pytest.approx(rh) == rh_round_trip


@given(
    floats(-50, 50),
    floats(-50, 50),
    floats(200, 400),
    floats(0.00001, 0.01),
    floats(100, 110),
    floats(-100, -50),
    integers(0, 10),
    sampled_from([tuple, list]),
)
def test_basis_tranformations(u, v, t, q, dp, dz, num_extra, container):
    args = tuple(
        [tf.convert_to_tensor(val) for val in [u, v, t, q, dp, dz] + [0.0] * num_extra]
    )
    orig = SpecificHumidityBasis(container(args))
    roundtrip = orig.to_rh().to_q()

    assert len(orig.args) == len(roundtrip.args)
    for k, (a, b) in enumerate(zip(orig.args, roundtrip.args)):
        assert pytest.approx(a.numpy()) == b.numpy(), k


def test_vertical_thickness_nonpositive(state):
    all_positive = (state["vertical_thickness_of_atmospheric_layer"] <= 0).all().item()
    assert all_positive
