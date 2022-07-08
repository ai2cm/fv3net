import numpy as np
import tensorflow as tf
from fv3fit.emulation.flux import TendencyToFlux, MoistStaticEnergyTransform


def _get_vertical_flux_transform():
    delp = tf.convert_to_tensor([1.0, 1, 2])
    interface_flux = tf.convert_to_tensor([0.5, 1, 2])
    down_sfc_flux = tf.convert_to_tensor([5.0])
    up_sfc_flux = tf.convert_to_tensor([1.5])
    x = {
        "delp": delp,
        "flux": interface_flux,
        "sfc_down": down_sfc_flux,
        "sfc_up": up_sfc_flux,
        "toa_net": interface_flux[0],
    }
    transform = TendencyToFlux(
        "tendency",
        "flux",
        "sfc_down",
        "sfc_up",
        "delp",
        net_toa_flux="toa_net",
        gravity=1.0,
    )
    expected_tendency = tf.convert_to_tensor([-0.5, -1, -0.75])
    return x, expected_tendency, transform


def test_TendencyToFlux_backward():
    x, expected_tendency, transform = _get_vertical_flux_transform()
    y = transform.backward(x)
    tf.debugging.assert_equal(y["tendency"], expected_tendency)
    for name in x:
        tf.debugging.assert_equal(x[name], y[name])


def test_TendencyToFlux_round_trip():
    x, _, transform = _get_vertical_flux_transform()
    y = transform.backward(x)
    x_round_tripped = transform.forward(y)
    tf.debugging.assert_equal(x_round_tripped["flux"], x["flux"])
    tf.debugging.assert_equal(x_round_tripped["sfc_down"], x["sfc_down"])


def test_TendencyToFlux_backward_names():
    transform = TendencyToFlux("a", "b", "c", "d", "e", "f")
    expected_requested_names = {"a", "d", "e", "f"}
    requested_names = transform.backward_names({"b", "c"})
    assert expected_requested_names == requested_names


def test_MoistStaticEnergyTransform_round_trip():
    heating = tf.convert_to_tensor([1.0, 1, 2])
    moistening = tf.convert_to_tensor([0.5, 1, 2])
    x = {"Q1": heating, "Q2": moistening}
    transform = MoistStaticEnergyTransform("Q1", "Q2", "Qm")
    x_round_tripped = transform.backward(transform.forward(x))
    np.testing.assert_allclose(x_round_tripped["Q1"], x["Q1"], rtol=1e-5, atol=1e-3)
    np.testing.assert_allclose(x_round_tripped["Q2"], x["Q2"])
    assert "Qm" in x_round_tripped
