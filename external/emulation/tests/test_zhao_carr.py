import numpy as np
from emulation.zhao_carr import (
    _limit_net_condensation_conserving,
    Input,
    ice_water_flag,
    latent_heat_phase_dependent,
)


def test__limit_net_condenstation():
    qv = np.array([[1, 1, 1], [0, 0, 0]], dtype=np.float64)
    qc = np.array([[0, 0, 0], [1, 1, 0]], dtype=np.float64)
    net_condensation = np.array([[1.5, 0.5, 0], [-1.5, -0.5, 0]], dtype=np.float64)
    expected_net_condensation = np.array([[1, 0.5, 0], [-1, -0.5, 0]])

    state = {
        Input.cloud_water: qc,
        Input.humidity: qv,
    }
    result = _limit_net_condensation_conserving(state, net_condensation)
    np.testing.assert_array_equal(result, expected_net_condensation)


def test_ice_water_flag_simple():
    temperature = np.array([[10, 0, -10, -15, -16]])
    cloud = np.array([[0, 0, 0, 1, 0]])
    iw = ice_water_flag(temperature_celsius=temperature, cloud=cloud)
    expected = np.array([[0, 0, 0.0, 1.0, 1.0]])
    np.testing.assert_array_equal(expected, iw)


def test_ice_water_flag_no_cloud_history():
    temperature = np.array([[-14, -16]])
    cloud = np.array([[0, 0]])
    iw = ice_water_flag(temperature_celsius=temperature, cloud=cloud)
    expected = np.array([[0, 1.0]])
    np.testing.assert_array_equal(expected, iw)


def test_latent_heat_phase_dependent(regtest):
    ans = latent_heat_phase_dependent(0.5)
    print(ans, file=regtest)
