import numpy as np
import scipy.linalg
import pytest

from fv3fit.reservoir.config import ReadoutHyperparameters
from fv3fit.reservoir.readout import (
    ReservoirComputingReadout,
    CombinedReservoirComputingReadout,
    square_even_terms,
    _sort_subdirs_numerically,
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


def test_readout_square_half_hidden_state():
    readout = ReservoirComputingReadout(
        hyperparameters=ReadoutHyperparameters(
            linear_regressor_kwargs={}, square_half_hidden_state=True
        ),
        coefficients=np.ones(shape=(2, 5)),
        intercepts=np.zeros(2),
    )
    input = np.array([2, 2, 2, 2, 2])
    # square even entries -> [4, 2, 4, 2, 4]
    output = readout.predict(input)
    np.testing.assert_array_almost_equal(output, np.array([16.0, 16.0]))


def test_reservoir_computing_readout_fit():
    unsquared = ReservoirComputingReadout(
        ReadoutHyperparameters(linear_regressor_kwargs={})
    )
    squared = ReservoirComputingReadout(
        ReadoutHyperparameters(
            linear_regressor_kwargs={}, square_half_hidden_state=True
        )
    )

    res_states = np.arange(10).reshape(2, 5)
    output_states = np.arange(6).reshape(2, 3)

    unsquared.fit(res_states, output_states)
    squared.fit(res_states, output_states)

    assert not np.allclose(unsquared.coefficients, squared.coefficients)


def test_combined_readout():
    state_size = 3
    output_size = 2
    readout_1 = ReservoirComputingReadout(
        hyperparameters=ReadoutHyperparameters(
            linear_regressor_kwargs={}, square_half_hidden_state=False
        ),
        coefficients=np.ones(shape=(output_size, state_size)),
        intercepts=np.zeros(output_size),
    )
    readout_2 = ReservoirComputingReadout(
        hyperparameters=ReadoutHyperparameters(
            linear_regressor_kwargs={}, square_half_hidden_state=False
        ),
        coefficients=2.0 * np.ones(shape=(output_size, state_size)),
        intercepts=np.zeros(output_size),
    )
    input = np.array([1, 1, 1])
    output_1 = readout_1.predict(input)
    output_2 = readout_2.predict(input)

    combined_readout = CombinedReservoirComputingReadout(
        readouts=[readout_1, readout_2]
    )
    combined_input = np.concatenate([input, input])
    output_combined = combined_readout.predict(combined_input)

    np.testing.assert_array_almost_equal(
        output_combined, np.concatenate([output_1, output_2])
    )


def test_combined_readout_inconsistent_hyperparameters():
    readout_1 = ReservoirComputingReadout(
        hyperparameters=ReadoutHyperparameters(
            linear_regressor_kwargs={}, square_half_hidden_state=True
        ),
        coefficients=np.ones(shape=(2, 2)),
        intercepts=np.zeros(2),
    )
    readout_2 = ReservoirComputingReadout(
        hyperparameters=ReadoutHyperparameters(
            linear_regressor_kwargs={}, square_half_hidden_state=False
        ),
        coefficients=np.ones(shape=(2, 2)),
        intercepts=np.zeros(2),
    )
    with pytest.raises(ValueError):
        CombinedReservoirComputingReadout(readouts=[readout_1, readout_2])


def test__sort_subdirs_numerically():
    subdirs = [
        "subdir_0",
        "subdir_1",
        "subdir_10",
        "subdir_11",
        "subdir_2",
    ]
    assert _sort_subdirs_numerically(subdirs) == [
        "subdir_0",
        "subdir_1",
        "subdir_2",
        "subdir_10",
        "subdir_11",
    ]


def test_combined_load(tmpdir):
    readouts, readout_paths = [], []
    coef_shape = (2, 2)
    output_size = 2
    for i in range(3):
        output_path = f"{str(tmpdir)}/readout_{i}"
        readout = ReservoirComputingReadout(
            hyperparameters=ReadoutHyperparameters(
                linear_regressor_kwargs={}, square_half_hidden_state=True
            ),
            coefficients=np.ones(coef_shape) * i,
            intercepts=np.zeros(output_size),
        )
        readout.dump(output_path)
        readouts.append(readout)
        readout_paths.append(output_path)
    combined_readout = CombinedReservoirComputingReadout.load(str(tmpdir))
    np.testing.assert_array_almost_equal(
        scipy.linalg.block_diag(*[r.coefficients for r in readouts]),
        combined_readout.coefficients.todense(),
    )
    np.testing.assert_array_almost_equal(
        np.concatenate([r.intercepts for r in readouts]), combined_readout.intercepts
    )
