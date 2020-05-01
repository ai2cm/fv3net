import numpy as np
from zarr_to_test_schema import *

import pytest


def test_sample_middle_dim():
    arr = np.ones((3, 4, 5))
    expected = sample(arr, sample_axes=[1], num_samples=2)
    assert expected.shape == (2, 3, 5)


def test_sample_multiple_sample_axes():
    arr = np.ones((3, 4, 5))
    expected = sample(arr, sample_axes=[1, 2], num_samples=2)
    assert expected.shape == (2, 3)


def test_sample_multiple_sample_axes_reorder():
    arr = np.ones((4, 5, 3))
    expected = sample(arr, sample_axes=[0, 1], num_samples=2)
    assert expected.shape == (2, 3)


def test_generate_array():
    r = Range(0, 1)
    a = Array(shape=(1,2), dtype=np.float32)
    out = a.generate(r)

    assert out.dtype == a.dtype
    assert out.shape == a.shape


def test_generate_chunked_array():
    r = Range(0, 1)
    a = ChunkedArray(shape=(10,10), dtype=np.float32, chunks=((5, 5), (5,5)))
    out = a.generate(r)

    assert out.dtype == a.dtype
    assert out.shape == a.shape
    assert out.chunks == a.chunks