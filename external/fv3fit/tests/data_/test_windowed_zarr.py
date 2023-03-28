from fv3fit.data import WindowedZarrLoader, VariableConfig
from fv3fit.data.tfdataset import get_n_windows
import tempfile
import xarray as xr
import numpy as np
import pytest
import contextlib
import cftime
from datetime import datetime, timedelta

NX, NY, NZ, NT = 5, 5, 8, 40


@contextlib.contextmanager
def temporary_zarr_path():
    dir = tempfile.TemporaryDirectory()

    ds = xr.Dataset(
        data_vars={
            "a": xr.DataArray(
                np.random.randn(NT, NX, NY, NZ), dims=["time", "x", "y", "z"]
            ),
            "a_sfc": xr.DataArray(np.random.randn(NT, NX, NY), dims=["time", "x", "y"]),
            "b": xr.DataArray(
                np.random.randn(NT, NX, NY, NZ), dims=["time", "x", "y", "z"]
            ),
            "c": xr.DataArray(
                np.random.randn(NT, NX, NY, NZ), dims=["time", "x", "y", "z"]
            ),
        }
    )
    ds.to_zarr(dir.name)
    yield dir.name


@pytest.mark.parametrize(
    "variable_names",
    [pytest.param(["a"], id="one_var"), pytest.param(["b", "c"], id="two_vars")],
)
def test_loader_gets_requested_variables(variable_names: str):
    with temporary_zarr_path() as data_path:
        loader = WindowedZarrLoader(
            data_path=data_path,
            window_size=10,
            unstacked_dims=["time", "z"],
            default_variable_config=VariableConfig(times="window"),
            variable_configs={},
        )
        dataset = loader.open_tfdataset(
            local_download_path=None, variable_names=variable_names
        )
        item = next(iter(dataset))
        assert set(item.keys()) == set(variable_names)


def test_loader_stacks_default_config():
    variable_names = ["a", "a_sfc"]
    with temporary_zarr_path() as data_path:
        loader = WindowedZarrLoader(
            data_path=data_path,
            window_size=10,
            unstacked_dims=["time", "z"],
            default_variable_config=VariableConfig(times="window"),
            variable_configs={},
        )
        dataset = loader.open_tfdataset(
            local_download_path=None, variable_names=variable_names
        ).unbatch()
        item = next(iter(dataset))
        # check that the given sample only contains the requested
        # unstacked_dims, and that they have the right lengths
        assert item["a"].shape[0] == 10
        assert item["a"].shape[1] == NZ
        assert len(item["a"].shape) == 2
        assert item["a_sfc"].shape[0] == 10
        assert len(item["a_sfc"].shape) == 1


def test_loader_stacks_default_config_without_stacked_dims():
    """
    Special case where all dimensions are included in unstacked_dims, relevant
    because a "sample" dimension may or may not be created in this case.
    """
    variable_names = ["a", "a_sfc"]
    window_size = 10
    with temporary_zarr_path() as data_path:
        loader = WindowedZarrLoader(
            data_path=data_path,
            window_size=10,
            unstacked_dims=["time", "x", "y", "z"],
            default_variable_config=VariableConfig(times="window"),
            variable_configs={},
        )
        dataset = loader.open_tfdataset(
            local_download_path=None, variable_names=variable_names
        )
        item = next(iter(dataset))
        assert item["a"].shape == [1, window_size, NX, NY, NZ]
        assert item["a_sfc"].shape == [1, window_size, NX, NY]


def test_loader_handles_window_start():
    """
    Special case where all dimensions are included in unstacked_dims, relevant
    because a "sample" dimension may or may not be created in this case.
    """
    variable_names = ["a", "a_sfc"]
    window_size = 10
    with temporary_zarr_path() as data_path:
        loader = WindowedZarrLoader(
            data_path=data_path,
            window_size=window_size,
            unstacked_dims=["time", "x", "y", "z"],
            default_variable_config=VariableConfig(times="window"),
            variable_configs={"a_sfc": VariableConfig(times="start")},
        )
        dataset = loader.open_tfdataset(
            local_download_path=None, variable_names=variable_names
        )
        item = next(iter(dataset))
        assert item["a"].shape == [1, window_size, NX, NY, NZ]
        assert item["a_sfc"].shape == [1, NX, NY]


def test_loader_handles_time_range():
    """
    Test that time_start_index and time_end_index are used to select the time
    """
    variable_names = ["a", "a_sfc"]
    window_size = 10
    with temporary_zarr_path() as data_path:
        ds = xr.open_zarr(data_path)
        loader = WindowedZarrLoader(
            data_path=data_path,
            window_size=window_size,
            unstacked_dims=["time", "x", "y", "z"],
            default_variable_config=VariableConfig(times="window"),
            variable_configs={"a_sfc": VariableConfig(times="start")},
            time_start_index=5,
            time_end_index=5 + window_size + 1,
        )
        dataset = loader.open_tfdataset(
            local_download_path=None, variable_names=variable_names
        )
        # result is a dictionary of Tensor
        item_tensors = next(iter(dataset))
        # retrieve numpy array
        item = item_tensors["a"].numpy()
        # remove inserted batch dimension when comparing
        item = item[0, :]
        expected = ds["a"].isel(time=slice(5, 15)).values
        np.testing.assert_array_equal(item, expected)


@pytest.mark.parametrize(
    ["n_times", "window_size", "n_windows"],
    [
        pytest.param(9, 10, 0, id="no_window"),
        pytest.param(10, 10, 1, id="one_window"),
        pytest.param(10, 6, 1, id="one_window_almost_two"),
        # two windows: [0, 1, 2, 3, 4, 5], [5, 6, 7, 8, 9, 10]
        pytest.param(11, 6, 2, id="two_windows"),
        pytest.param(12, 6, 2, id="two_windows_one_extra_point"),
    ],
)
def test_get_n_windows(n_times, window_size, n_windows):
    assert get_n_windows(n_times, window_size) == n_windows


def test_variable_config_stacks_requested_order():
    config = VariableConfig(times="start")
    ds = xr.Dataset(
        data_vars={
            "a": (("time", "r", "q", "t", "s"), np.zeros((1, 2, 3, 4, 5))),
            "time": (("time",), [datetime.now()]),
        }
    )
    result = config.get_record("a", ds=ds, unstacked_dims=["r", "q", "t", "s"])
    assert result.shape == (1, 2, 3, 4, 5)
    result = config.get_record("a", ds=ds, unstacked_dims=["s", "t", "q", "r"])
    assert result.shape == (1, 5, 4, 3, 2)


def test_loader_decodes_time_from_cftime():
    dir = tempfile.TemporaryDirectory()
    ds = xr.Dataset(
        data_vars={
            "a": xr.DataArray(
                np.random.randn(NT, NX, NY, NZ), dims=["time", "x", "y", "z"]
            ),
        },
        coords={
            "time": xr.DataArray(
                [
                    cftime.DatetimeJulian(2018, 1, 1) + timedelta(days=i)
                    for i in range(NT)
                ],
                dims=["time"],
            ),
        },
    )
    ds.to_zarr(dir.name)
    loader = WindowedZarrLoader(
        data_path=dir.name,
        window_size=10,
        unstacked_dims=["time", "x", "y", "z"],
        default_variable_config=VariableConfig(times="window"),
        variable_configs={},
    )
    dataset = loader.open_tfdataset(
        local_download_path=None, variable_names=["a", "time"]
    )
    item_tensors = next(iter(dataset))
    a = item_tensors["a"].numpy()
    time = item_tensors["time"].numpy()
    assert time.shape[0] == a.shape[0]  # sample within batch
    assert time.shape[1] == a.shape[1]  # window size
    assert time[0, 1] - time[0, 0] == 60 * 60 * 24  # one day
    assert len(time.shape) == 2


def test_loader_decodes_time_from_datetime64():
    dir = tempfile.TemporaryDirectory()
    ds = xr.Dataset(
        data_vars={
            "a": xr.DataArray(
                np.random.randn(NT, NX, NY, NZ), dims=["time", "x", "y", "z"]
            ),
        },
        coords={
            "time": xr.DataArray(
                np.arange(NT), dims=["time"], attrs={"units": "days since 2018-01-01"}
            ),
        },
    )
    ds.to_zarr(dir.name)
    loader = WindowedZarrLoader(
        data_path=dir.name,
        window_size=10,
        unstacked_dims=["time", "x", "y", "z"],
        default_variable_config=VariableConfig(times="window"),
        variable_configs={},
    )
    dataset = loader.open_tfdataset(
        local_download_path=None, variable_names=["a", "time"]
    )
    item_tensors = next(iter(dataset))
    a = item_tensors["a"].numpy()
    time = item_tensors["time"].numpy()
    assert time.shape[0] == a.shape[0]  # sample within batch
    assert time.shape[1] == a.shape[1]  # window size
    assert len(time.shape) == 2
    assert time[0, 1] - time[0, 0] == 60 * 60 * 24  # one day
