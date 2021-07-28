import pytest
import os
import tempfile
import numpy as np
import xarray as xr

from fv3fit.emulation.data import io


def _get_ds():

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


def test_get_nc_files():

    xr_dataset = _get_ds()

    with tempfile.TemporaryDirectory() as tmpdir:
        num_files = 3
        orig_paths = [os.path.join(tmpdir, f"file{i}.nc") for i in range(num_files)]
        for path in orig_paths:
            xr_dataset.to_netcdf(path)

        result_files = io.get_nc_files(tmpdir)
        assert len(result_files) == num_files
        for path in result_files:
            assert path in orig_paths


class MockGCSFilesystem:

    protocol = ("gs", "gcs")

    def glob(*args):
        return [
            "fake-bucket/file1.nc",
            "fake-bucket/file2.nc",
        ]


@pytest.mark.network
def test_get_nc_files_remote_protocol_prepend():

    fs = MockGCSFilesystem()
    result_files = io.get_nc_files("gs://fake-bucket", fs=fs)

    assert len(result_files) == 2
    for path in result_files:
        assert path.startswith("gs://")
