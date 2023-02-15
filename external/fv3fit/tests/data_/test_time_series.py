import contextlib
import numpy as np
import pytest
import tempfile
import xarray as xr

from fv3fit.data import ReservoirTimeSeriesLoader


NX, NY, NZ, NT = 5, 6, 8, 10


@contextlib.contextmanager
def temporary_netcdfs_dir():
    dir = tempfile.TemporaryDirectory()

    # Test datasets that have different number of timesteps
    ds = xr.Dataset(
        data_vars={
            "a_sfc": xr.DataArray(np.random.randn(NX, NY, NT), dims=["x", "y", "time"]),
            "b": xr.DataArray(
                np.random.randn(NX, NY, NZ, NT), dims=["x", "y", "z", "time"]
            ),
        }
    )

    ds.to_netcdf(f"{dir.name}/0.nc")
    ds.isel(time=slice(0, 2)).to_netcdf(f"{dir.name}/1.nc")
    yield dir.name


@pytest.mark.parametrize(
    "dim_order, expected_shape_first_batch",
    [
        (["time", "x", "y", "z"], (NT, NX, NY, NZ)),
        (["time", "y", "z", "x"], (NT, NY, NZ, NX)),
    ],
)
def test_ReservoirTimeSeriesLoader(dim_order, expected_shape_first_batch):
    with temporary_netcdfs_dir() as data_path:
        loader = ReservoirTimeSeriesLoader(data_path=data_path, dim_order=dim_order)
        dataset = loader.open_tfdataset(
            local_download_path=None, variable_names=["a_sfc", "b"]
        )
        first_batch = next(iter(dataset))
        assert first_batch["b"].shape == expected_shape_first_batch


def test_error_on_time_not_first():
    with pytest.raises(ValueError):
        with temporary_netcdfs_dir() as data_path:
            loader = ReservoirTimeSeriesLoader(
                data_path=data_path, dim_order=["z", "time", "x", "y"]
            )
            loader.open_tfdataset(
                local_download_path=None, variable_names=["a_sfc", "b"]
            )
