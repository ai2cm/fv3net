import os
import tempfile
from distutils import dir_util

import numpy as np
import pytest
import xarray as xr

from ._fine_res import generate_fine_res
from ._nudging import generate_nudging
from .core import Range, generate, load

timestep1 = "20160801.001500"
timestep2 = "20160801.003000"
times_centered_str = [timestep1, timestep2]


@pytest.fixture(scope="module")
def dataset_fixtures_dir(tmpdir_factory, request):
    """Creates a temporary directory for the contents of the
    synth datasets for use in dataset fixtures, and returns its path"""

    test_dir, _ = os.path.splitext(__file__)

    tmpdir = tmpdir_factory.mktemp("pytest_data")

    if os.path.isdir(test_dir):
        dir_util.copy_tree(test_dir, str(tmpdir))

    return tmpdir


@pytest.fixture(params=["nudging_tendencies", "fine_res_apparent_sources"])
def data_source_name(request):
    return request.param


@pytest.fixture(scope="module")
def nudging_dataset_path():
    with tempfile.TemporaryDirectory() as nudging_dir:
        generate_nudging(nudging_dir)
        yield nudging_dir


@pytest.fixture(scope="module")
def fine_res_dataset_path():
    with tempfile.TemporaryDirectory() as fine_res_dir:
        generate_fine_res(fine_res_dir, times_centered_str)
        yield fine_res_dir


@pytest.fixture
def data_source_path(dataset_fixtures_dir, data_source_name):
    with tempfile.TemporaryDirectory() as data_dir:
        if data_source_name == "nudging_tendencies":
            generate_nudging(data_dir)
        elif data_source_name == "fine_res_apparent_sources":
            generate_fine_res(data_dir, times_centered_str)
        else:
            raise NotImplementedError()
        yield data_dir


@pytest.fixture(scope="module")
def C48_SHiELD_diags_dataset_path(dataset_fixtures_dir):

    with tempfile.TemporaryDirectory() as C48_SHiELD_diags_dir:
        C48_SHiELD_diags_zarrpath = _generate_C48_SHiELD_diags_dataset(
            dataset_fixtures_dir, C48_SHiELD_diags_dir
        )
        yield C48_SHiELD_diags_zarrpath


def _generate_C48_SHiELD_diags_dataset(datadir, C48_SHiELD_diags_dir):

    with open(str(datadir.join("C48_SHiELD_diags.json"))) as f:
        C48_SHiELD_diags_schema = load(f)
    C48_SHiELD_diags_zarrpath = os.path.join(
        C48_SHiELD_diags_dir, "gfsphysics_15min_coarse.zarr"
    )
    C48_SHiELD_diags_dataset = generate(C48_SHiELD_diags_schema)
    C48_SHiELD_diags_dataset_1 = C48_SHiELD_diags_dataset.assign_coords(
        {"time": [timestep1]}
    )
    C48_SHiELD_diags_dataset_2 = C48_SHiELD_diags_dataset.assign_coords(
        {"time": [timestep2]}
    )
    C48_SHiELD_diags_dataset = xr.concat(
        [C48_SHiELD_diags_dataset_1, C48_SHiELD_diags_dataset_2], dim="time"
    )
    C48_SHiELD_diags_dataset.to_zarr(C48_SHiELD_diags_zarrpath, consolidated=True)
    return C48_SHiELD_diags_zarrpath


@pytest.fixture
def grid_dataset(dataset_fixtures_dir):
    random = np.random.RandomState(0)
    with open(str(dataset_fixtures_dir.join("grid_schema.json"))) as f:
        grid_schema = load(f)
    grid_ranges = {"area": Range(1, 2)}
    grid = generate(grid_schema, ranges=grid_ranges).load()
    grid["land_sea_mask"][:] = random.choice(
        [0, 1, 2], size=grid["land_sea_mask"].shape
    )
    return grid


@pytest.fixture
def grid_dataset_path(grid_dataset):
    with tempfile.TemporaryDirectory() as grid_dir:
        grid_path = os.path.join(grid_dir, "grid.nc")
        grid_dataset.to_netcdf(grid_path)
        yield grid_path
