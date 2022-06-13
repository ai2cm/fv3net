import mappm
import numpy as np


def test_mappm():
    p_in = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])[None, :]
    f_in = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0])[None, :]
    p_out = np.asarray([0.5, 1.2, 2.4, 2.8, 3.2, 4.5])[None, :]
    n_columns = 1.0
    iv = 1.0
    kord = 1.0
    dummy_ptop = 0.0
    result = mappm.mappm(p_in, f_in, p_out, 1, n_columns, iv, kord, dummy_ptop)
    expected = np.asarray([[0.35, 1.3, 2.1, 2.5, 3.35]], dtype=np.float32)
    # input was identity, so output should be also
    np.testing.assert_almost_equal(result, expected, decimal=5)


def test_mappm_out_of_bounds():
    p_in = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0])[None, :]
    f_in = np.asarray([1.5, 2.5, 3.5, 4.5])[None, :]
    p_out = np.asarray([0.0, 2.5, 3.5, 4.5, 50.0])[None, :]
    n_columns = 1.0
    iv = 1.0
    kord = 1.0
    dummy_ptop = 0.0
    result = mappm.mappm(p_in, f_in, p_out, 1, n_columns, iv, kord, dummy_ptop)
    expected = np.asarray([[1.5, 3.0, 4.0, 4.502747]], dtype=np.float32)
    # input was identity, so output should be also
    np.testing.assert_almost_equal(result, expected, decimal=5)


def test_mappm_nans():
    p_in = np.asarray([1.0, 2.0, 3.0, 2.0, 5.0])[None, :]
    f_in = np.asarray([np.nan, np.nan, np.nan, np.nan])[None, :]
    p_out = np.asarray([0.0, 2.5, 3.5, 4.5, 50.0])[None, :]
    n_columns = 1.0
    iv = 1.0
    kord = 1.0
    dummy_ptop = 0.0
    result = mappm.mappm(p_in, f_in, p_out, 1, n_columns, iv, kord, dummy_ptop)
    expected = np.asarray([[np.nan, np.nan, np.nan, np.nan]], dtype=np.float32)
    # input was identity, so output should be also
    np.testing.assert_almost_equal(result, expected)
