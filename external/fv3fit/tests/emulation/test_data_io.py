import os
import tempfile
import numpy as np
import xarray as xr

from fv3fit.emulation.data import io


def test_get_nc_files():

    xr_dataset = xr.Dataset(
        {
            "air_temperature": xr.DataArray(
                data=np.arange(30).reshape(10, 3), dims=["sample", "z"]
            ),
            "specific_humidity": xr.DataArray(
                data=np.arange(30, 60).reshape(10, 3), dims=["sample", "z"]
            ),
        }
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        num_files = 3
        paths = [os.path.join(tmpdir, f"file{i}.nc") for i in range(num_files)]
        for path in paths:
            xr_dataset.to_netcdf(path)

        result_files = io.get_nc_files(tmpdir)
        assert len(result_files) == num_files
        for path in paths:
            assert path in result_files
