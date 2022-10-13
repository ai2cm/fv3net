import numpy as np
import pytest
from scipy import sparse

from sklearn.linear_model import Ridge

from fv3fit.reservoir.readout import (
    ReservoirComputingReadout,
    square_even_terms,
)


@pytest.mark.parametrize(
    "arr, axis, expected",
    [
        (np.arange(4), 0, np.array([0, 1, 4, 3])),
        (np.arange(4).reshape(1, -1), 1, np.array([[0, 1, 4, 3]])),
        (np.arange(8).reshape(2, 4), 0, np.array([[0, 1, 4, 9], [4, 5, 6, 7]])),
        (
            np.arange(10).reshape(2, 5),
            1,
            np.array([[0, 1, 4, 3, 16], [25, 6, 49, 8, 81]]),
        ),
    ],
)
def test_square_even_terms(arr, axis, expected):
    np.testing.assert_array_equal(square_even_terms(arr, axis=axis), expected)


class MultiOutputMeanRegressor:
    def __init__(self, n_outputs: int):
        self.n_outputs = n_outputs

    def predict(self, input):
        # returns vector of size n_outputs, with each element
        # the mean of the input vector elements
        return np.full(self.n_outputs, np.mean(input))


def _sparse_allclose(A, B, atol=1e-8):
    # https://stackoverflow.com/a/47771340
    r1, c1, v1 = sparse.find(A)
    r2, c2, v2 = sparse.find(B)
    index_match = np.array_equal(r1, r2) & np.array_equal(c1, c2)
    if index_match == 0:
        return False
    else:
        return np.allclose(v1, v2, atol=atol)


def test_readout_square_half_hidden_state():
    readout = ReservoirComputingReadout(
        linear_regressor=MultiOutputMeanRegressor(n_outputs=2),
        square_half_hidden_state=True,
    )
    input = np.array([2, 2, 2, 2, 2])
    # square even entries -> [4, 2, 4, 2, 4]
    output = readout.predict(input)
    np.testing.assert_array_almost_equal(output, np.array([16.0 / 5, 16.0 / 5]))


def test_reservoir_computing_readout_fit():
    unsquared = ReservoirComputingReadout(Ridge())
    squared = ReservoirComputingReadout(Ridge(), square_half_hidden_state=True)

    res_states = np.arange(10).reshape(2, 5)
    output_states = np.arange(6).reshape(2, 3)

    unsquared.fit(res_states, output_states)
    squared.fit(res_states, output_states)

    assert not np.allclose(
        unsquared.linear_regressor.coef_, squared.linear_regressor.coef_
    )
