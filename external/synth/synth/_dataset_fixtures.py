import os
import tempfile
import numpy as np
import xarray as xr
import pytest
from distutils import dir_util

from .core import load, generate, Range


timestep1 = "20160801.001500"
timestep1_npdatetime_fmt = "2016-08-01T00:15:00"
timestep2 = "20160801.003000"
timestep2_npdatetime_fmt = "2016-08-01T00:30:00"


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
def nudging_dataset_path(dataset_fixtures_dir):

    with tempfile.TemporaryDirectory() as nudging_dir:
        _generate_nudging_dataset(dataset_fixtures_dir, nudging_dir)
        yield nudging_dir


def _generate_nudging_dataset(datadir, nudging_dir):

    nudging_after_dynamics_zarrpath = os.path.join(
        nudging_dir, "outdir-3h", "after_dynamics.zarr"
    )
    with open(str(datadir.join("after_dynamics.json"))) as f:
        nudging_after_dynamics_schema = load(f)
    nudging_after_dynamics_dataset = generate(
        nudging_after_dynamics_schema
    ).assign_coords(
        {
            "time": [
                np.datetime64(timestep1_npdatetime_fmt),
                np.datetime64(timestep2_npdatetime_fmt),
            ]
        }
    )
    nudging_after_dynamics_dataset.to_zarr(
        nudging_after_dynamics_zarrpath, consolidated=True
    )

    nudging_after_physics_zarrpath = os.path.join(
        nudging_dir, "outdir-3h", "after_physics.zarr"
    )
    with open(str(datadir.join("after_physics.json"))) as f:
        nudging_after_physics_schema = load(f)
    nudging_after_physics_dataset = generate(
        nudging_after_physics_schema
    ).assign_coords(
        {
            "time": [
                np.datetime64(timestep1_npdatetime_fmt),
                np.datetime64(timestep2_npdatetime_fmt),
            ]
        }
    )
    nudging_after_physics_dataset.to_zarr(
        nudging_after_physics_zarrpath, consolidated=True
    )

    nudging_tendencies_zarrpath = os.path.join(
        nudging_dir, "outdir-3h", "nudging_tendencies.zarr"
    )
    with open(str(datadir.join("nudging_tendencies.json"))) as f:
        nudging_tendencies_schema = load(f)
    nudging_tendencies_dataset = generate(nudging_tendencies_schema).assign_coords(
        {
            "time": [
                np.datetime64(timestep1_npdatetime_fmt),
                np.datetime64(timestep2_npdatetime_fmt),
            ]
        }
    )
    nudging_tendencies_dataset.to_zarr(nudging_tendencies_zarrpath, consolidated=True)


@pytest.fixture(scope="module")
def fine_res_dataset_path(dataset_fixtures_dir):

    with tempfile.TemporaryDirectory() as fine_res_dir:
        fine_res_zarrpath = _generate_fine_res_dataset(
            dataset_fixtures_dir, fine_res_dir
        )
        yield fine_res_zarrpath


def _generate_fine_res_dataset(datadir, fine_res_dir):
    """ Note that this does not follow the pattern of the other two datasets
    in that the synthetic data are not stored in the original format of the
    fine res data (tiled netcdfs), but instead as a zarr, because synth does
    not currently support generating netcdfs or splitting by tile
    """

    fine_res_zarrpath = os.path.join(fine_res_dir, "fine_res_budget.zarr")
    with open(str(datadir.join("fine_res_budget.json"))) as f:
        fine_res_budget_schema = load(f)
    fine_res_budget_dataset = generate(fine_res_budget_schema)
    fine_res_budget_dataset_1 = fine_res_budget_dataset.assign_coords(
        {"time": [timestep1]}
    )
    fine_res_budget_dataset_2 = fine_res_budget_dataset.assign_coords(
        {"time": [timestep2]}
    )
    fine_res_budget_dataset = xr.concat(
        [fine_res_budget_dataset_1, fine_res_budget_dataset_2], dim="time"
    )
    fine_res_budget_dataset.to_zarr(fine_res_zarrpath, consolidated=True)

    return fine_res_zarrpath


@pytest.fixture
def data_source_path(dataset_fixtures_dir, data_source_name):
    with tempfile.TemporaryDirectory() as data_dir:
        if data_source_name == "one_step_tendencies":
            _generate_one_step_dataset(dataset_fixtures_dir, data_dir)
            data_source_path = data_dir
        elif data_source_name == "nudging_tendencies":
            _generate_nudging_dataset(dataset_fixtures_dir, data_dir)
            data_source_path = data_dir
        elif data_source_name == "fine_res_apparent_sources":
            fine_res_zarrpath = _generate_fine_res_dataset(
                dataset_fixtures_dir, data_dir
            )
            data_source_path = fine_res_zarrpath
        else:
            raise NotImplementedError()
        yield data_source_path


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
