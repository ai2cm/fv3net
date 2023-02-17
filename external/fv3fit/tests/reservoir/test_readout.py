import numpy as np
import scipy.linalg
import pytest

from fv3fit.reservoir.config import (
    ReadoutHyperparameters,
    BatchLinearRegressorHyperparameters,
)
from fv3fit.reservoir.readout import (
    ReservoirComputingReadout,
    CombinedReservoirComputingReadout,
    square_even_terms,
    _sort_subdirs_numerically,
    BatchLinearRegressor,
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
            linear_regressor_config=BatchLinearRegressorHyperparameters(l2=0),
            square_half_hidden_state=True,
        ),
        coefficients=np.ones(shape=(5, 2)),
        intercepts=np.zeros(2),
    )
    input = np.array([2, 2, 2, 2, 2])
    # square even entries -> [4, 2, 4, 2, 4]
    output = readout.predict(input)
    np.testing.assert_array_almost_equal(output, np.array([16.0, 16.0]))


def test_reservoir_computing_readout_fit():
    lr_config = BatchLinearRegressorHyperparameters(l2=0)
    unsquared = ReservoirComputingReadout(
        ReadoutHyperparameters(linear_regressor_config=lr_config)
    )
    squared = ReservoirComputingReadout(
        ReadoutHyperparameters(
            linear_regressor_config=lr_config, square_half_hidden_state=True
        )
    )

    res_states = np.random.rand(20, 5)
    output_states = np.random.rand(20, 3)

    unsquared.fit(res_states, output_states)
    squared.fit(res_states, output_states)

    assert not np.allclose(unsquared.coefficients, squared.coefficients)


def test_combined_readout():
    np.random.seed(0)

    state_size = 3
    output_size = 2
    lr_config = BatchLinearRegressorHyperparameters(l2=0)
    readout_1 = ReservoirComputingReadout(
        hyperparameters=ReadoutHyperparameters(
            linear_regressor_config=lr_config, square_half_hidden_state=False
        ),
        coefficients=np.random.rand(state_size, output_size),
        intercepts=np.random.rand(output_size),
    )
    readout_2 = ReservoirComputingReadout(
        hyperparameters=ReadoutHyperparameters(
            linear_regressor_config=lr_config, square_half_hidden_state=False
        ),
        coefficients=np.random.rand(state_size, output_size),
        intercepts=np.random.rand(output_size),
    )
    input = np.array([[1, 1, 1], [2, 2, 2]])
    output_1 = readout_1.predict(input)
    output_2 = readout_2.predict(input)

    combined_readout = CombinedReservoirComputingReadout(
        readouts=[readout_1, readout_2]
    )
    combined_input = np.concatenate([input, input], axis=1)
    output_combined = combined_readout.predict(combined_input)
    np.testing.assert_array_almost_equal(
        output_combined, np.concatenate([output_1, output_2], axis=1)
    )


def test_combined_readout_inconsistent_hyperparameters():
    lr_config = BatchLinearRegressorHyperparameters(l2=0)

    readout_1 = ReservoirComputingReadout(
        hyperparameters=ReadoutHyperparameters(
            linear_regressor_config=lr_config, square_half_hidden_state=True
        ),
        coefficients=np.ones(shape=(2, 2)),
        intercepts=np.zeros(2),
    )
    readout_2 = ReservoirComputingReadout(
        hyperparameters=ReadoutHyperparameters(
            linear_regressor_config=lr_config, square_half_hidden_state=False
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
    np.random.seed(0)
    readouts, readout_paths = [], []
    coef_shape = (2, 2)
    output_size = 2
    for i in range(3):
        output_path = f"{str(tmpdir)}/readout_{i}"
        readout = ReservoirComputingReadout(
            hyperparameters=ReadoutHyperparameters(
                linear_regressor_config=BatchLinearRegressorHyperparameters(l2=0),
                square_half_hidden_state=True,
            ),
            coefficients=np.random.rand(*coef_shape),
            intercepts=np.random.rand(output_size),
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


def test_BatchLinearRegressor():
    # y0 = 1*x0 + 2*x1 + 4*x3 + 3
    # y1 = 1*x0 + 2*x1 + 4*x3 - 1
    X = np.array([[1, 2, 1], [2, 3, 0]])
    y = np.array([[12.0, 8], [11, 7]])

    # use_least_squares_solve needed when overdetermined test cases
    # result in nonsingular XT.X
    config = BatchLinearRegressorHyperparameters(
        l2=0, add_bias_term=True, use_least_squares_solve=True
    )
    lr = BatchLinearRegressor(config)
    lr.batch_update(X, y)

    coefficients, intercepts = lr.get_weights()
    np.testing.assert_array_almost_equal(np.dot(X, coefficients) + intercepts, y)


def test_BatchLinearRegressor_iterative_result_same_as_one_shot():
    x_dim, y_dim, N = 25, 5, 50
    l2 = 0.1

    W_truth = np.random.rand(y_dim, x_dim + 1)
    noise = np.random.normal(-0.1, 0.1, size=(N, y_dim))

    n_batches = 5
    X_batches = [
        np.concatenate([np.random.rand(N, x_dim), np.ones((N, 1))], axis=1)
        for i in range(n_batches)
    ]
    y_batches = [np.dot(Xb, W_truth.T) + noise for Xb in X_batches]

    X_full = np.concatenate(X_batches, axis=0)
    y_full = np.concatenate(y_batches, axis=0)

    # use_least_squares_solve needed when overdetermined test cases
    # result in nonsingular XT.X
    config = BatchLinearRegressorHyperparameters(
        l2=l2, add_bias_term=True, use_least_squares_solve=True
    )
    batch_lr = BatchLinearRegressor(config)
    for Xb, yb in zip(X_batches, y_batches):
        batch_lr.batch_update(Xb, yb)
    coefficients, intercepts = batch_lr.get_weights()

    full_lr = BatchLinearRegressor(config)
    full_lr.batch_update(X_full, y_full)
    coefficients_full, intercepts_full = batch_lr.get_weights()

    np.testing.assert_array_almost_equal(coefficients, coefficients_full)
    np.testing.assert_array_almost_equal(intercepts, intercepts_full)


def test_BatchLinearRegressor_add_bias_term():
    # y0 = 1*x0 + 2*x1 + 4*x3 + 3
    # y1 = 1*x0 + 2*x1 + 4*x3 - 1
    X = np.array([[1, 2, 1], [2, 3, 0]])
    X_with_bias_const = np.array([[1, 2, 1, 1], [2, 3, 0, 1]])
    y = np.array([[12.0, 8], [11, 7]])

    # use_least_squares_solve needed when overdetermined test cases
    # result in nonsingular XT.X
    config_add_bias = BatchLinearRegressorHyperparameters(
        l2=0, add_bias_term=True, use_least_squares_solve=True
    )
    config_no_bias = BatchLinearRegressorHyperparameters(
        l2=0, add_bias_term=False, use_least_squares_solve=True
    )
    lr_add_bias = BatchLinearRegressor(config_add_bias)
    lr_no_bias = BatchLinearRegressor(config_no_bias)

    lr_add_bias.batch_update(X, y)
    lr_no_bias.batch_update(X_with_bias_const, y)

    coefficients_add_bias, intercepts_add_bias = lr_add_bias.get_weights()
    coefficients_no_bias, intercepts_no_bias = lr_no_bias.get_weights()

    np.testing.assert_array_equal(coefficients_add_bias, coefficients_no_bias)
    np.testing.assert_array_equal(intercepts_add_bias, intercepts_no_bias)
