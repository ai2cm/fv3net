import numpy as np
import os
import pytest

from fv3fit.reservoir.config import BatchLinearRegressorHyperparameters
from fv3fit.reservoir.readout import (
    ReservoirComputingReadout,
    combine_readouts,
    BatchLinearRegressor,
)


def test_readout_dump_load(tmpdir):
    state_size = 4
    output_size = 2

    readout = ReservoirComputingReadout(
        coefficients=np.random.rand(state_size, output_size),
        intercepts=np.random.rand(output_size),
    )
    output_path = os.path.join(tmpdir, "readout")
    readout.dump(output_path)
    loaded_readout = ReservoirComputingReadout.load(output_path)

    x = np.ones(state_size)
    np.testing.assert_array_almost_equal(
        loaded_readout.predict(x), readout.predict(x),
    )


def test_combine_readouts():
    np.random.seed(0)

    state_size = 3
    output_size = 2
    readout_1 = ReservoirComputingReadout(
        coefficients=np.random.rand(state_size, output_size),
        intercepts=np.random.rand(output_size),
    )
    readout_2 = ReservoirComputingReadout(
        coefficients=np.random.rand(state_size, output_size),
        intercepts=np.random.rand(output_size),
    )
    input = np.array([[1, 1, 1], [2, 2, 2]])
    output_1 = readout_1.predict(input)
    output_2 = readout_2.predict(input)

    combined_readout = combine_readouts(readouts=[readout_1, readout_2])
    combined_input = np.concatenate([input, input], axis=1)
    output_combined = combined_readout.predict(combined_input)
    np.testing.assert_array_almost_equal(
        output_combined, np.concatenate([output_1, output_2], axis=1)
    )


def test_BatchLinearRegressor():
    # y0 = 1*x0 + 2*x1 + 4*x3 + 3
    # y1 = 1*x0 + 2*x1 + 4*x3 - 1
    X = np.array([[1, 2, 1], [2, 3, 0]])
    y = np.array([[12.0, 8], [11, 7]])

    # use_least_squares_solve needed when underdetermined test cases
    # result in singular XT.X
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

    # use_least_squares_solve needed when underdetermined test cases
    # result in singular XT.X
    config = BatchLinearRegressorHyperparameters(
        l2=l2, add_bias_term=True, use_least_squares_solve=True
    )
    batch_lr = BatchLinearRegressor(config)
    for Xb, yb in zip(X_batches, y_batches):
        batch_lr.batch_update(Xb, yb)
    coefficients, intercepts = batch_lr.get_weights()

    full_lr = BatchLinearRegressor(config)
    full_lr.batch_update(X_full, y_full)
    coefficients_full, intercepts_full = full_lr.get_weights()

    np.testing.assert_array_almost_equal(coefficients, coefficients_full)
    np.testing.assert_array_almost_equal(intercepts, intercepts_full)


def test_BatchLinearRegressor_add_bias_term():
    # y0 = 1*x0 + 2*x1 + 4*x3 + 3
    # y1 = 1*x0 + 2*x1 + 4*x3 - 1
    X = np.array([[1, 2, 1], [2, 3, 0]])
    X_with_bias_const = np.array([[1, 2, 1, 1], [2, 3, 0, 1]])
    y = np.array([[12.0, 8], [11, 7]])

    # use_least_squares_solve needed when underdetermined test cases
    # result in singular XT.X
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


def test_BatchLinearRegressor_error_on_missing_bias_col():
    X = np.array([[1, 2, 1, 10], [2, 3, 0, 2]])
    X_with_bias_const = np.array([[1, 2, 1, 1], [2, 3, 0, 1]])

    y = np.array([[12.0, 8], [11, 7]])

    config_no_bias = BatchLinearRegressorHyperparameters(
        l2=0, add_bias_term=False, use_least_squares_solve=True
    )
    lr_no_bias = BatchLinearRegressor(config_no_bias)

    # should pass check if last col is constant
    lr_no_bias.batch_update(X_with_bias_const, y)

    # fail if add_bias_term is False but last col is not constant
    with pytest.raises(ValueError):
        lr_no_bias.batch_update(X, y)
