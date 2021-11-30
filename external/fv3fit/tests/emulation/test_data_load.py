import os
import pytest
import numpy as np
import tensorflow as tf
import xarray as xr

from fv3fit.emulation.data import load, TransformConfig, netcdf_url_to_dataset


def _get_dataset() -> xr.Dataset:
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


@pytest.fixture
def config():
    return TransformConfig(
        input_variables=["air_temperature"],
        output_variables=["specific_humidity"],
        use_tensors=True,
    )


@pytest.fixture(scope="module")
def nc_dir(tmp_path_factory):

    ds = _get_dataset()
    netcdf_dir = tmp_path_factory.mktemp("netcdf_files")

    nfiles = 10
    for i in range(nfiles):
        filename = os.path.join(netcdf_dir, f"file{i:02d}.nc")
        ds = ds + i
        ds.to_netcdf(filename)

    return netcdf_dir


@pytest.fixture
def nc_dir_files(nc_dir):

    return [str(f) for f in nc_dir.glob("*.nc")]


def test__seq_to_tf_dataset():

    batches = [np.arange(30).reshape(10, 3)] * 3

    def transform(batch):
        return batch * 2

    tf_ds = load._seq_to_tf_dataset(batches, transform)
    assert isinstance(tf_ds, tf.data.Dataset)

    result = next(tf_ds.batch(10).as_numpy_iterator())
    np.testing.assert_equal(result, batches[0] * 2)


def test_netcdf_directory_to_tf_dataset(config, nc_dir):

    tf_ds = load.nc_dir_to_tf_dataset(str(nc_dir), config)

    assert isinstance(tf_ds, tf.data.Dataset)
    tensor_ins, tensor_outs = next(iter(tf_ds.batch(150)))  # larger than total samples
    assert len(tensor_ins[0]) == 100  # nfiles * sample size
    assert len(tensor_outs[0]) == 100
    assert isinstance(tensor_ins[0], tf.Tensor)
    assert isinstance(tensor_outs[0], tf.Tensor)


def test_netcdf_files_to_tf_dataset(config, nc_dir_files):

    tf_ds = load.nc_files_to_tf_dataset(nc_dir_files, config)

    assert isinstance(tf_ds, tf.data.Dataset)
    tensor_ins, tensor_outs = next(iter(tf_ds.batch(150)))  # larger than total samples
    assert len(tensor_ins[0]) == 100  # nfiles * sample size
    assert len(tensor_outs[0]) == 100
    assert isinstance(tensor_ins[0], tf.Tensor)
    assert isinstance(tensor_outs[0], tf.Tensor)


def test_netcdf_dir_to_tf_dataset_with_nfiles(config, nc_dir):
    ds = load.nc_dir_to_tf_dataset(str(nc_dir), config, nfiles=1)
    (tensor_in,), _ = next(iter(ds.batch(30)))

    assert len(tensor_in) == 10  # only a single file


def test_netcdf_dir_to_tf_dataset_with_shuffle(config, nc_dir):

    # seeds that won't have same first batch
    random1 = np.random.RandomState(10)
    random2 = np.random.RandomState(20)

    ds1 = load.nc_dir_to_tf_dataset(
        str(nc_dir), config, shuffle=True, random_state=random1
    )
    ds2 = load.nc_dir_to_tf_dataset(
        str(nc_dir), config, shuffle=True, random_state=random2
    )
    (tensor_in1,), _ = next(iter(ds1.batch(10)))
    (tensor_in2,), _ = next(iter(ds2.batch(10)))

    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(tensor_in1, tensor_in2)


def test_batches_to_tf_dataset(config):
    xr_dataset = _get_dataset()
    batches = [xr_dataset] * 3
    tf_ds = load.batches_to_tf_dataset(batches, config)
    assert isinstance(tf_ds, tf.data.Dataset)


def test_netcdf_url_to_dataset(tmpdir):
    nfiles = 3
    ds = _get_dataset()
    for f in [tmpdir.join(f"file{i}.nc") for i in range(nfiles)]:
        ds.to_netcdf(str(f))

    tf_ds = netcdf_url_to_dataset(str(tmpdir), variables=set(ds))

    assert len(tf_ds) == nfiles
    for item in tf_ds:
        for variable in ds:
            assert isinstance(item[variable], tf.Tensor)
