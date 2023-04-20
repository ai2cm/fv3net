from pathlib import Path

import fv3fit.data
import numpy as np
import vcm
import contextlib
import pytest
import tempfile
import xarray as xr
import tensorflow as tf
from fv3fit.data.netcdf.load import _numerical_sort_names


def test_NCDirLoader(tmp_path: Path):
    path = tmp_path / "a.nc"
    cache = tmp_path / ".cache"
    ds = vcm.cdl_to_dataset(
        """
    netcdf A {
        dimensions:
            sample = 4;
            z = 1;
        variables:
            float a(sample, z);
        data:
            a = 0, 1, 2, 3;
    }
    """
    )

    ds.to_netcdf(path.as_posix())
    loader = fv3fit.data.NCDirLoader(tmp_path.as_posix())
    tfds = loader.open_tfdataset(cache.as_posix(), ["a"])
    for data in tfds.as_numpy_iterator():
        a = data["a"]
    np.testing.assert_array_equal(ds["a"], a)


def test_Netcdf_from_dict():
    fv3fit.data.NCDirLoader.from_dict({"url": "some/path"})


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
def test_NCDirLoader_dim_order(dim_order, expected_shape_first_batch):
    with temporary_netcdfs_dir() as data_path:
        loader = fv3fit.data.NCDirLoader(
            url=data_path, dim_order=dim_order, shuffle=False, varying_first_dim=True,
        )
        dataset = loader.open_tfdataset(
            local_download_path=None, variable_names=["a_sfc", "b"]
        )
        first_batch = next(iter(dataset))
        assert first_batch["b"].shape == expected_shape_first_batch


def test_NCDirLoader_varying_first_dim_allowed():
    with temporary_netcdfs_dir() as data_path:
        loader = fv3fit.data.NCDirLoader(
            url=data_path,
            dim_order=["time", "x", "y", "z"],
            shuffle=False,
            varying_first_dim=True,
        )
        dataset = loader.open_tfdataset(
            local_download_path=None, variable_names=["a_sfc", "b"]
        )
        first_dim_sizes = []
        for batch in dataset:
            first_dim_sizes.append(batch["b"].shape[0])
        assert first_dim_sizes[0] != first_dim_sizes[1]


def test_NCDirLoader_varying_first_dim_not_allowed():
    with temporary_netcdfs_dir() as data_path:
        loader = fv3fit.data.NCDirLoader(
            url=data_path,
            dim_order=["time", "x", "y", "z"],
            shuffle=False,
            varying_first_dim=False,
        )
        dataset = loader.open_tfdataset(
            local_download_path=None, variable_names=["a_sfc", "b"]
        )
        with pytest.raises(tf.errors.InvalidArgumentError):
            for batch in dataset:
                _ = batch["b"]


def test_error_missing_data_dim_in_specified_order():
    with temporary_netcdfs_dir() as data_path:
        loader = fv3fit.data.NCDirLoader(
            url=data_path,
            dim_order=["time", "x", "y"],
            shuffle=False,
            varying_first_dim=True,
        )
        with pytest.raises(ValueError):
            loader.open_tfdataset(
                local_download_path=None, variable_names=["a_sfc", "b"]
            )


@pytest.mark.parametrize(
    "names, sorted_names",
    [
        (["c.nc", "a.nc", "b.nc"], ["a.nc", "b.nc", "c.nc"]),
        (
            ["3.nc", "0.nc", "1.nc", "10.nc", "2.nc"],
            ["0.nc", "1.nc", "2.nc", "3.nc", "10.nc"],
        ),
        (
            ["t_0.nc", "t_1.nc", "t_10.nc", "t_2.nc"],
            ["t_0.nc", "t_1.nc", "t_2.nc", "t_10.nc"],
        ),
        (["b_0_c_1", "b_0_c_0", "a_1_b_1"], ["a_1_b_1", "b_0_c_0", "b_0_c_1"]),
    ],
)
def test_sort_netcdfs(names, sorted_names):
    _numerical_sort_names(names)
    assert names == sorted_names
