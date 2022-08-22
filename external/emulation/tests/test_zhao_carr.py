import numpy as np

from emulation.zhao_carr import _limit_net_condensation_conserving, Input


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
