import os
import tempfile
from distutils import dir_util

import numpy as np
import pytest

from ._fine_res import generate_fine_res
from ._nudging import generate_nudging
from .core import Range, generate, load

timestep1 = "20160801.001500"
timestep1_npdatetime_fmt = "2016-08-01T00:15:00"
timestep2 = "20160801.003000"
timestep2_npdatetime_fmt = "2016-08-01T00:30:00"

times_numpy = [
    np.datetime64(timestep1_npdatetime_fmt),
    np.datetime64(timestep2_npdatetime_fmt),
]
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


@pytest.fixture(
    params=["one_step_tendencies", "nudging_tendencies", "fine_res_apparent_sources"]
)
def data_source_name(request):
    return request.param


@pytest.fixture(scope="module")
def one_step_dataset_path(dataset_fixtures_dir):

    with tempfile.TemporaryDirectory() as one_step_dir:
        _generate_one_step_dataset(dataset_fixtures_dir, one_step_dir)
        yield one_step_dir


def _generate_one_step_dataset(datadir, one_step_dir):

    with open(str(datadir.join("one_step.json"))) as f:
        one_step_schema = load(f)
    one_step_dataset = generate(one_step_schema)
    one_step_dataset_1 = one_step_dataset.assign_coords({"initial_time": [timestep1]})
    one_step_dataset_2 = one_step_dataset.assign_coords({"initial_time": [timestep2]})
    one_step_dataset_1.to_zarr(
        os.path.join(one_step_dir, f"{timestep1}.zarr"), consolidated=True,
    )
    one_step_dataset_2.to_zarr(
        os.path.join(one_step_dir, f"{timestep2}.zarr"), consolidated=True,
    )


@pytest.fixture(scope="module")
def nudging_dataset_path():
    with tempfile.TemporaryDirectory() as nudging_dir:
        generate_nudging(nudging_dir, times_numpy)
        yield nudging_dir


@pytest.fixture(scope="module")
def fine_res_dataset_path():
    with tempfile.TemporaryDirectory() as fine_res_dir:
        generate_fine_res(fine_res_dir, times_centered_str)
        yield fine_res_dir


@pytest.fixture
def data_source_path(dataset_fixtures_dir, data_source_name):
    with tempfile.TemporaryDirectory() as data_dir:
        if data_source_name == "one_step_tendencies":
            _generate_one_step_dataset(dataset_fixtures_dir, data_dir)
        elif data_source_name == "nudging_tendencies":
            generate_nudging(data_dir, times_numpy)
        elif data_source_name == "fine_res_apparent_sources":
            generate_fine_res(data_dir, times_centered_str)
        else:
            raise NotImplementedError()
        yield data_dir


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
