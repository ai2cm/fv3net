import pytest
import tensorflow as tf
from fv3fit.emulation.thermo import conservative_precipitation_zhao_carr


def test_conservative_precipitation():

    nz = 5
    expected_precip = 1.2

    one = tf.ones((1, nz))

    precip = conservative_precipitation_zhao_carr(
        one, one, one, one - expected_precip / nz, mass=1
    )
    assert precip.numpy() == pytest.approx(expected_precip / 1000)
