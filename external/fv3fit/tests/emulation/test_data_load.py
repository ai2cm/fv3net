import os
import tempfile
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
def xr_dataset():
    return _get_dataset()


@pytest.fixture
def config():
    return TransformConfig(
        input_variables=["air_temperature"],
        output_variables=["specific_humidity"],
        use_tensors=True,
    )


def test__seq_to_tf_dataset():

    batches = [np.arange(30).reshape(10, 3)] * 3

    def transform(batch):
        return batch * 2

    tf_ds = load._seq_to_tf_dataset(batches, transform)
    assert isinstance(tf_ds, tf.data.Dataset)

    result = next(tf_ds.batch(10).as_numpy_iterator())
    np.testing.assert_equal(result, batches[0] * 2)


def test_nc_to_tf_dataset(xr_dataset, config):

    with tempfile.TemporaryDirectory() as tmpdir:
        files = [os.path.join(tmpdir, f"file{i}.nc") for i in range(3)]
        for f in files:
            xr_dataset.to_netcdf(f)

        tf_ds = load.nc_files_to_tf_dataset(files, config)
        assert isinstance(tf_ds, tf.data.Dataset)

        tf_ds = load.nc_dir_to_tf_dataset(tmpdir, config)
        assert isinstance(tf_ds, tf.data.Dataset)


def test_batches_to_tf_dataset(xr_dataset, config):

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
