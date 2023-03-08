import numpy as np
import emulation.zhao_carr as zc
from emulation.zhao_carr import (
    _limit_net_condensation_conserving,
    Input,
    ice_water_flag,
    latent_heat_phase_dependent,
    _strict_conservative_precip_from_TOA_to_surface,
    mass_to_mixing_ratio,
    mixing_ratio_to_mass,
    liquid_water_equivalent,
)


def test__limit_net_condensation():
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


def test__strict_conservative_precip_limiting():
    """
    Test that negative sources/sinks are fixed and that availability limts
    are obeyed.
    """

    # Production - Evaporation example
    # TOP   3 - 2
    # MID   0 - 0
    # BOT   1 - 2
    c_to_p = np.array([[1.0], [-2.0], [3.0]])
    p_to_v = np.array([[4.0], [-1.0], [2.0]])

    new_c_to_p, new_p_to_v, total_p = _strict_conservative_precip_from_TOA_to_surface(
        c_to_p, p_to_v
    )

    np.testing.assert_equal(new_c_to_p, [[1.0], [0.0], [3.0]])
    np.testing.assert_equal(new_p_to_v, [[2.0], [0.0], [2.0]])
    np.testing.assert_equal(total_p, np.zeros_like(total_p))


def test_mass_mixing_ratio_conversions(regtest):
    a = np.array(2.0)
    delp = np.array(1.0)

    mass = mixing_ratio_to_mass(a, delp)
    mixing_ratio = mass_to_mixing_ratio(mass, delp)

    np.testing.assert_equal(a, mixing_ratio)

    print(mass, file=regtest)


def test_liquid_water_equivalent(regtest):
    water_m = liquid_water_equivalent(np.array(10.0))
    print(water_m, file=regtest)


def _create_states():
    shp = (5, 10)

    state = {
        zc.GscondOutput.cloud_water: np.ones(shp) * 4,
        zc.GscondOutput.humidity: np.ones(shp),
        zc.GscondOutput.temperature: np.ones(shp) * 10,
        zc.Input.delp: np.ones(shp),
    }

    emulator = {
        zc.PrecpdOutput.cloud_water: np.ones(shp) * 2,
        zc.PrecpdOutput.humidity: np.ones(shp) * 2,
    }

    return state, emulator


def test_enforce_conservative():

    state, emulator = _create_states()

    result = zc.enforce_conservative_precpd(state, emulator)
    assert zc.PrecpdOutput.precip in result
    assert zc.PrecpdOutput.temperature in result


def test_enforce_conservative_overwrite():

    state, _ = _create_states()

    # adjust emulated state to data which should be adjusted / overwritten
    dummy_data = -1 * np.ones_like(state[zc.GscondOutput.humidity])
    emulator = {
        zc.PrecpdOutput.cloud_water: dummy_data * -10,  # negative precip source
        zc.PrecpdOutput.humidity: dummy_data,
        zc.PrecpdOutput.temperature: dummy_data,
        zc.PrecpdOutput.precip: dummy_data,
    }

    result = zc.enforce_conservative_precpd(state, emulator)

    for v in result.values():
        assert not np.any(v == -1)

    assert not np.any(result[zc.PrecpdOutput.cloud_water] == 10)


def test_simple_conservative():

    state, emulator = _create_states()
    result = zc.conservative_precip_simple(state, emulator)
    assert zc.PrecpdOutput.precip in result

    precip = result[zc.PrecpdOutput.precip]
    result[zc.PrecpdOutput.precip] = -1 * np.ones_like(precip)
    overwritten = zc.conservative_precip_simple(state, result)

    assert not np.any(overwritten[zc.PrecpdOutput.precip] == -1)
