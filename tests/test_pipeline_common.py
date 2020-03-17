import pytest
import tempfile
import pathlib
import os
import re
from datetime import timedelta

from vcm.cubedsphere.constants import TIME_FMT
from vcm import parse_datetime_from_str
from fv3net.pipelines.common import (
    list_timesteps,
    get_alphanumeric_unique_tag,
    subsample_timesteps_at_interval,
    get_base_timestep_interval,
)


@pytest.fixture
def fv3_timesteps():
    """Generate a list of 12 15-minute timesteps"""

    initial_step = "20160801.001500"
    initial_datetime = parse_datetime_from_str(initial_step)
    delta = timedelta(minutes=15)
    timesteps = []
    for i in range(12):
        curr_dt = initial_datetime + i * delta
        timesteps.append(curr_dt.strftime(TIME_FMT))

    return tuple(timesteps)


@pytest.fixture
def timestep_dir(fv3_timesteps):

    timesteps = list(fv3_timesteps[:3])
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


def test_alphanumeric_unique_tag_length():

    tlen = 8
    tag = get_alphanumeric_unique_tag(tlen)
    assert len(tag) == tlen

    with pytest.raises(ValueError):
        get_alphanumeric_unique_tag(0)


def test_alphanumeric_uniq_tag_is_lowercase_alphanumeric():
    """
    Generate a really long tag to be reasonably certain character restrictions
    are enforced.
    """

    tag = get_alphanumeric_unique_tag(250)
    pattern = "^[a-z0-9]+$"
    res = re.match(pattern, tag)
    assert res is not None


def test_get_base_timestep_interval_bad_threshold(fv3_timesteps):

    with pytest.raises(ValueError):
        get_base_timestep_interval(fv3_timesteps, vote_threshold=0.5)

    with pytest.raises(ValueError):
        get_base_timestep_interval(fv3_timesteps, vote_threshold=1.1)


def test_get_base_timestep_interval_bad_num_to_check(fv3_timesteps):

    with pytest.raises(ValueError):
        get_base_timestep_interval(fv3_timesteps[:3], num_to_check=4)


def test_get_base_timestep_interval_no_majority(fv3_timesteps):
    """
    Create timesteps list with all different intervals in between, should
    not be able to determine a majority choice for a base interval
    """
    missing_timesteps = list(fv3_timesteps)[:8]
    for i in [5, 4, 2]:
        del missing_timesteps[i]

    with pytest.raises(ValueError):
        get_base_timestep_interval(
            missing_timesteps, num_to_check=4, vote_threshold=0.51
        )


def test_get_base_timestep_interval(fv3_timesteps):
    timesteps = list(fv3_timesteps[:6])
    # create 75% majority of
    del timesteps[3]

    res = get_base_timestep_interval(timesteps, num_to_check=4, vote_threshold=0.75)
    assert res == timedelta(minutes=15)


def test_subsample_timesteps_at_interval(fv3_timesteps):

    # assumes 15 minute timesteps for tests below
    timesteps = list(fv3_timesteps[:4])

    assert subsample_timesteps_at_interval(timesteps, 5) == timesteps
    assert subsample_timesteps_at_interval(timesteps, 30) == timesteps[::2]

    with pytest.raises(ValueError):
        # frequency larger than available times
        subsample_timesteps_at_interval(timesteps, 60)

    with pytest.raises(ValueError):
        # frequency not aligned
        subsample_timesteps_at_interval(timesteps, 7)


def test_subsample_timesteps_at_interval_with_pairs(fv3_timesteps):

    timesteps = list(fv3_timesteps[:7])
    # remove paired timestep of first entry
    del timesteps[1]

    # subsample at 30-minute interval with pairs
    subsampled = subsample_timesteps_at_interval(timesteps, 30, paired_steps=True)
    # First and last timestep should not be included since there's no associated pair
    assert subsampled == timesteps[2:-1]
