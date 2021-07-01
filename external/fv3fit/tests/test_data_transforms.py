from typing import Mapping
import pytest
import numpy as np
import tensorflow as tf
import xarray as xr
from fv3fit.emulation.data import transforms as xfm


@pytest.fixture
def xr_dataset():
    return xr.Dataset({
        "air_temperature": xr.DataArray(data=np.arange(30).reshape(10, 3), dims=["sample", "z"]),
        "specific_humidity": xr.DataArray(data=np.arange(30, 60).reshape(10, 3), dims=["sample", "z"])
    })


def test_xr_dataset_to_ndarray_dataset(xr_dataset):

    result = xfm.to_ndarrays(xr_dataset)
    assert len(result) == len(xr_dataset)

    for key, data in xr_dataset.items():
        assert key in result
        result_data = result[key]
        assert isinstance(result_data, np.ndarray)
        np.testing.assert_equal(data.values, result_data)


def test_xr_dataset_to_tensor_dataset(xr_dataset):
    result = xfm.to_tensors(xr_dataset)
    assert len(result) == len(xr_dataset)

    for key, data in xr_dataset.items():
        assert key in result
        result_data = result[key]
        assert isinstance(result_data, tf.Tensor)
        result_data = tf.convert_to_tensor(data.values)


def test_select_antarcic():
    pass


def test_select_antarctic_no_overlap():
    pass


def test_select_antarctic_no_latitude_in_dataset():
    pass


def test_group_input_output():
    pass


def test_group_input_output_empty_var_list():
    pass


def test_maybe_subselect():
    # Parametrize for all potential datasets
    pass


def test_maybe_subselect_empty_selection_map():
    pass


def test_maybe_expand_feature_dim():
    # parametrize ndarrays and tensors
    # have 1d and 2d variable
    pass
