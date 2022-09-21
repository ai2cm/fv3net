from emulation import zhao_carr
import numpy as np


def test_ice_water_flag_simple():
    temperature = np.array([[10, 0, -10, -15, -16]])
    cloud = np.array([[0, 0, 0, 1, 0]])
    iw = zhao_carr.ice_water_flag(temperature_celsius=temperature, cloud=cloud)
    expected = np.array([[0, 0, 0.0, 1.0, 1.0]])
    np.testing.assert_array_equal(expected, iw)


def test_ice_water_flag_no_cloud_history():
    temperature = np.array([[-14, -16]])
    cloud = np.array([[0, 0]])
    iw = zhao_carr.ice_water_flag(temperature_celsius=temperature, cloud=cloud)
    expected = np.array([[0, 1.0]])
    np.testing.assert_array_equal(expected, iw)


def test_latent_heat_phase_dependent(regtest):
    ans = zhao_carr.latent_heat_phase_dependent(0.5)
    print(ans, file=regtest)
