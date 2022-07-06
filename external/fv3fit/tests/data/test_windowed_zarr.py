from fv3fit.data import WindowedZarrLoader, VariableConfig
import tempfile
import xarray as xr
import numpy as np
import pytest
import contextlib

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
            batch_size=1,
        )
        dataset = loader.get_data(
            local_download_path=None, variable_names=variable_names
        )
        item = next(iter(dataset))
        assert set(item.keys()) == set(variable_names)


def test_loader_stacks_default_config():
    variable_names = ["a", "a_sfc"]
    batch_size = 1
    with temporary_zarr_path() as data_path:
        loader = WindowedZarrLoader(
            data_path=data_path,
            window_size=10,
            unstacked_dims=["time", "z"],
            default_variable_config=VariableConfig(times="window"),
            variable_configs={},
            batch_size=batch_size,
        )
        dataset = loader.get_data(
            local_download_path=None, variable_names=variable_names
        )
        item = next(iter(dataset))
        assert item["a"].shape[0] == batch_size
        assert len(item["a"].shape) == 3
        assert item["a"].shape[-1] == NZ
        assert item["a_sfc"].shape[0] == batch_size
        assert len(item["a_sfc"].shape) == 2


def test_loader_stacks_default_config_without_stacked_dims():
    """
    Special case where all dimensions are included in unstacked_dims, relevant
    because a "sample" dimension may or may not be created in this case.
    """
    variable_names = ["a", "a_sfc"]
    batch_size = 1
    window_size = 10
    with temporary_zarr_path() as data_path:
        loader = WindowedZarrLoader(
            data_path=data_path,
            window_size=10,
            unstacked_dims=["time", "x", "y", "z"],
            default_variable_config=VariableConfig(times="window"),
            variable_configs={},
            batch_size=batch_size,
        )
        dataset = loader.get_data(
            local_download_path=None, variable_names=variable_names
        )
        item = next(iter(dataset))
        assert item["a"].shape == [batch_size, window_size, NX, NY, NZ]
        assert item["a_sfc"].shape == [batch_size, window_size, NX, NY]


def test_loader_handles_window_start():
    """
    Special case where all dimensions are included in unstacked_dims, relevant
    because a "sample" dimension may or may not be created in this case.
    """
    variable_names = ["a", "a_sfc"]
    batch_size = 1
    window_size = 10
    with temporary_zarr_path() as data_path:
        loader = WindowedZarrLoader(
            data_path=data_path,
            window_size=10,
            unstacked_dims=["time", "x", "y", "z"],
            default_variable_config=VariableConfig(times="window"),
            variable_configs={"a_sfc": VariableConfig(times="start")},
            batch_size=batch_size,
        )
        dataset = loader.get_data(
            local_download_path=None, variable_names=variable_names
        )
        item = next(iter(dataset))
        assert item["a"].shape == [batch_size, window_size, NX, NY, NZ]
        assert item["a_sfc"].shape == [batch_size, NX, NY]
