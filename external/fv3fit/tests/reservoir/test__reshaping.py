import numpy as np

from fv3fit.reservoir._reshaping import (
    flatten_2d_keeping_columns_contiguous,
    stack_data,
    split_1d_samples_into_2d_rows,
)


def test_stack_data_has_time_dim():
    time_series = np.array([np.ones((2, 2)) * i for i in range(10)])
    stacked = stack_data(time_series, keep_first_dim=True)
    assert stacked.shape == (10, 4)
    np.testing.assert_array_equal(stacked[-1], np.array([9, 9, 9, 9]))


def test_stack_data_no_time_dim():
    data = [
        np.arange(4).reshape(2, 2),
        -1 * np.arange(4).reshape(2, 2),
    ]
    stacked = stack_data(data, keep_first_dim=False)
    np.testing.assert_array_equal(stacked, np.array([0, 1, 2, 3, 0, -1, -2, -3]))


def test_flatten_2d_keeping_columns_contiguous():
    x = np.array([[1, 2], [3, 4], [5, 6]])
    np.testing.assert_array_equal(
        flatten_2d_keeping_columns_contiguous(x), np.array([1, 3, 5, 2, 4, 6])
    )


def test_split_1d_samples_into_2d_rows():
    x = np.arange(12)
    x_2d = split_1d_samples_into_2d_rows(x, n_rows=4, keep_first_dim_shape=False)
    np.testing.assert_array_equal(
        x_2d, np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
    )


def test_split_1d_samples_into_2d_rows_keep_first_dim_shape():
    nt = 3
    x = np.array([np.arange(12) for i in range(nt)])
    x_2d = split_1d_samples_into_2d_rows(x, n_rows=4, keep_first_dim_shape=True)

    expected = np.array(
        [np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]) for i in range(nt)]
    )

    np.testing.assert_array_equal(x_2d, expected)
