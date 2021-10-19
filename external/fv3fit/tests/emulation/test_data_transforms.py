import os
import pytest
import tempfile
import numpy as np
import tensorflow as tf
import xarray as xr
from fv3fit.emulation.data import transforms


@pytest.fixture
def xr_dataset():
    return xr.Dataset(
        {
            "air_temperature": xr.DataArray(
                data=np.arange(30).reshape(10, 3), dims=["sample", "z"]
            ),
            "specific_humidity": xr.DataArray(
                data=np.arange(30, 60).reshape(10, 3), dims=["sample", "z"]
            ),
        }
    )


def test_open_netcdf(xr_dataset: xr.Dataset):

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "data.nc")
        xr_dataset.to_netcdf(path)

        result = transforms.open_netcdf_dataset(path)
        assert isinstance(result, xr.Dataset)
        xr.testing.assert_equal(result, xr_dataset)


def test_xr_dataset_to_ndarray_dataset(xr_dataset):

    result = transforms.to_ndarrays(xr_dataset)
    assert len(result) == len(xr_dataset)

    for key, data in xr_dataset.items():
        assert key in result
        result_data = result[key]
        assert isinstance(result_data, np.ndarray)
        np.testing.assert_equal(data.values, result_data)


def test_xr_dataset_to_tensor_dataset(xr_dataset):
    result = transforms.to_tensors(xr_dataset, dtype=tf.float32)
    assert len(result) == len(xr_dataset)

    for key, data in xr_dataset.items():
        assert key in result
        result_data = result[key]
        assert isinstance(result_data, tf.Tensor)
        expected = tf.convert_to_tensor(data, dtype=tf.float32)
        tf.debugging.assert_equal(result_data, expected)


@pytest.mark.parametrize(
    "lats, data, expected",
    [
        ([-55, -60, -65], [1, 2, 3], [3]),
        ([55, 60, 65], [1, 2, 3], []),
        ([-61, -65, -80], [1, 2, 3], [1, 2, 3]),
    ],
)
def test_select_antarctic(lats, data, expected):
    lats_da = xr.DataArray(np.deg2rad(lats), dims=["sample"])
    data_da = xr.DataArray(data, dims=["sample"])
    dataset = xr.Dataset({"latitude": lats_da, "field": data_da})
    result = transforms.select_antarctic(dataset)

    expected_da = xr.DataArray(expected, dims=["sample"])
    xr.testing.assert_equal(expected_da, result["field"])


def test_group_input_output_input():

    dataset = {"a": np.array([0, 1]), "b": np.array([2, 3]), "c": np.array([4, 5])}

    inputs_, outputs_ = transforms.group_inputs_outputs(["a"], ["c", "b"], dataset)
    assert isinstance(inputs_, tuple)
    assert isinstance(outputs_, tuple)
    assert len(inputs_) == 1
    np.testing.assert_array_equal(inputs_[0], dataset["a"])
    assert len(outputs_) == 2
    np.testing.assert_array_equal(outputs_[0], dataset["c"])
    np.testing.assert_array_equal(outputs_[1], dataset["b"])


def test_group_input_output_empty_var_list():

    dataset = {"a": np.array([0, 1]), "b": np.array([2, 3]), "c": np.array([4, 5])}

    inputs_, outputs_ = transforms.group_inputs_outputs([], [], dataset)
    assert inputs_ == tuple()
    assert outputs_ == tuple()


@pytest.mark.parametrize(
    "dataset",
    [
        xr.Dataset({"X": (["sample", "feature"], np.arange(40).reshape(10, 4))}),
        {"X": np.arange(40).reshape(10, 4)},
        {"X": tf.convert_to_tensor(np.arange(40).reshape(10, 4))},
    ],
)
def test_maybe_subselect_feature_dim_dataset_inputs(dataset):

    subselect_map = {"X": slice(2, None)}

    result = transforms.maybe_subselect_feature_dim(subselect_map, dataset)
    assert len(result["X"].shape) == len(dataset["X"].shape)
    np.testing.assert_equal(
        result["X"], np.arange(40).reshape(10, 4)[..., slice(2, None)]
    )


def test_maybe_subselect_feature_dim_empty_selection_map():

    subselect_map = {}
    full_data = np.arange(40).reshape(10, 4)

    result = transforms.maybe_subselect_feature_dim(subselect_map, {"X": full_data})
    np.testing.assert_equal(result["X"], full_data)


@pytest.mark.parametrize(
    "dataset",
    [
        {"X": np.arange(40).reshape(10, 4), "y": np.arange(20)},
        {
            "X": tf.convert_to_tensor(np.arange(40).reshape(10, 4)),
            "y": tf.convert_to_tensor(np.arange(20)),
        },
    ],
)
def test_expand_single_dim_data(dataset):

    result = transforms.expand_single_dim_data(dataset)
    assert result["X"].shape == (10, 4)
    assert result["y"].shape == (20, 1)


def test_derived():

    ds = {
        "air_temperature_input": xr.DataArray(
            np.ones((10, 4)), dims=["sample", "feature"]
        ),
        "air_temperature_output": xr.DataArray(
            np.ones((10, 4)) * 3, dims=["sample", "feature"]
        ),
    }

    dT_name = "tendency_of_air_temperature_due_to_microphysics"
    all_vars = list(ds.keys()) + [dT_name]

    derived = transforms.derived_dataset(all_vars, ds, tendency_timestep_sec=2)
    assert "air_temperature_input" in derived
    assert "air_temperature_output" in derived
    assert dT_name in derived
    np.testing.assert_array_equal(derived[dT_name], np.ones((10, 4)))
