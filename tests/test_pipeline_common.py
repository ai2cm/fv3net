import pytest
import tempfile
import pathlib
import os
from fv3net.pipelines.common import (
    list_timesteps,
    get_unique_tag,
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


def test_unique_tag_length():

    tlen = 8
    tag = get_unique_tag(tlen)
    assert len(tag) == tlen

    with pytest.raises(ValueError):
        get_unique_tag(0)
