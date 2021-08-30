import tensorflow as tf
from fv3fit.emulation.thermobasis.thermo import SpecificHumidityBasis
import pytest

from hypothesis import given
from hypothesis.strategies import floats, integers


@given(
    floats(-50, 50),
    floats(-50, 50),
    floats(200, 400),
    floats(0.00001, 0.01),
    floats(100, 110),
    floats(-100, -50),
    integers(0, 10),
)
def test_basis_tranformations(u, v, t, q, dp, dz, num_extra):

    scalars = tf.convert_to_tensor([0.0]) * num_extra
    args = dict(u=u, v=v, T=t, q=q, dp=dp, dz=dz)
    args_tf = {key: tf.convert_to_tensor(val) for key, val in args.items()}
    orig = SpecificHumidityBasis(scalars=scalars, **args_tf)
    roundtrip = orig.to_rh().to_q()

    assert len(orig.args) == len(roundtrip.args)
    for k, (a, b) in enumerate(zip(orig.args, roundtrip.args)):
        assert pytest.approx(a.numpy()) == b.numpy(), k


def test_vertical_thickness_nonpositive(state):
    all_positive = (state["vertical_thickness_of_atmospheric_layer"] <= 0).all().item()
    assert all_positive
