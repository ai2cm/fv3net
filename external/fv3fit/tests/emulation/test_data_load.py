import os
import pytest
import tempfile
import numpy as np
import tensorflow as tf
import xarray as xr

from fv3fit.emulation.data import load


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


def test_get_nc_files(xr_dataset):

    with tempfile.TemporaryDirectory() as tmpdir:
        num_files = 3
        paths = [os.path.join(tmpdir, f"file{i}.nc") for i in range(num_files)]
        for path in paths:
            xr_dataset.to_netcdf(path)

        result_files = load.get_nc_files(tmpdir)
        assert len(result_files) == num_files
        for path in paths:
            assert path in result_files


def test_batched_to_tf_dataset():

    batches = [np.arange(30).reshape(10, 3)] * 3

    def transform(batch):
        return batch * 2

    tf_ds = load.batched_to_tf_dataset(batches, transform)
    assert isinstance(tf_ds, tf.data.Dataset)

    result = next(tf_ds.batch(10).as_numpy_iterator())
    np.testing.assert_equal(result, batches[0] * 2)
