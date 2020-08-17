import pytest
import tempfile
import pathlib
import os
from typing import Iterable
from fv3net.pipelines.common import (
    list_timesteps,
    subsample_timesteps_at_interval,
)


@pytest.fixture
def timestep_dir():

    timesteps = ["20160801.001500", "20160801.003000", "20160801.004500"]
    extra_file = "not_a_timestep.nc"
    extra_dir = "what_a_config_dir"

    with tempfile.TemporaryDirectory() as tmpdir:
        for out_dir in timesteps + [extra_dir]:
            os.makedirs(os.path.join(tmpdir, out_dir))

        pathlib.Path(tmpdir, extra_file).touch()

        yield tmpdir, timesteps


def test_timestep_lister(timestep_dir):

    tmpdir, timesteps = timestep_dir
    timesteps_found = list_timesteps(tmpdir)
    assert len(timesteps_found) == len(timesteps)
    for check_timestep in timesteps_found:
        assert check_timestep in timesteps


def test_timestep_lister_sorted(timestep_dir):

    tmpdir, timesteps = timestep_dir
    timesteps.sort()
    timesteps_found = list_timesteps(tmpdir)
    for i, ref_timestep in enumerate(timesteps):
        assert timesteps_found[i] == ref_timestep


def test_timestep_lister_nonexistent_dir():

    res = list_timesteps("/nonexistent/directory/")
    assert not res and isinstance(res, Iterable)


def test_subsample_timesteps_at_interval():

    timesteps = [
        "20160801.001500",
        "20160801.003000",
        "20160801.004500",
        "20160801.010000",
    ]

    assert subsample_timesteps_at_interval(timesteps, 5) == timesteps
    assert subsample_timesteps_at_interval(timesteps, 30) == timesteps[::2]

    with pytest.raises(ValueError):
        # frequency larger than available times
        subsample_timesteps_at_interval(timesteps, 60)

    with pytest.raises(ValueError):
        # frequency not aligned
        subsample_timesteps_at_interval(timesteps, 7)
