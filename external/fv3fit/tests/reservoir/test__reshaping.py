import numpy as np

from fv3fit.reservoir._reshaping import split_1d_samples_into_2d_rows


def test_split_1d_samples_into_2d_rows():
    x = np.arange(12)
    x_2d = split_1d_samples_into_2d_rows(x, n_rows=4)
    np.testing.assert_array_equal(
        x_2d, np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
    )
