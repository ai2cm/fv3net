import numpy as np
from fv3fit.reservoir._reshaping import flatten_2d_keeping_columns_contiguous


def test_flatten_2d_keeping_columns_contiguous():
    x = np.array([[1, 2], [3, 4], [5, 6]])
    np.testing.assert_array_equal(
        flatten_2d_keeping_columns_contiguous(x), np.array([1, 3, 5, 2, 4, 6])
    )
