import os
import tempfile
import pytest
import numpy as np
import tensorflow as tf
import xarray as xr

from fv3fit.emulation.data import load, TransformConfig


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


@pytest.fixture
def config():
    return TransformConfig(
        input_variables=["air_temperature"],
        output_variables=["specific_humidity"],
        use_tensors=True
    )


def test_seq_to_tf_dataset():

    batches = [np.arange(30).reshape(10, 3)] * 3

    def transform(batch):
        return batch * 2

    tf_ds = load.seq_to_tf_dataset(batches, transform)
    assert isinstance(tf_ds, tf.data.Dataset)

    result = next(tf_ds.batch(10).as_numpy_iterator())
    np.testing.assert_equal(result, batches[0] * 2)


def test_nc_files_to_tf_dataset(xr_dataset, config):

    with tempfile.TemporaryDirectory() as tmpdir:
        files = [os.path.join(tmpdir, f"file{i}.nc") for i in range(3)]
        for f in files:
            xr_dataset.to_netcdf(f)

        tf_ds = load.nc_files_to_tf_dataset(files, config)
        assert isinstance(tf_ds, tf.data.Dataset)


def test_batches_to_tf_dataset(xr_dataset, config):

    batches = [xr_dataset] * 3
    tf_ds = load.batches_to_tf_dataset(batches, config)
    assert isinstance(tf_ds, tf.data.Dataset)
