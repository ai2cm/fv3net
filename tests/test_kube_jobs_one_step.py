import pytest
import os
import pathlib
import tempfile
import shutil

from fv3net.pipelines.kube_jobs import one_step

PWD = pathlib.Path(os.path.abspath(__file__)).parent
TEST_DATA_DIR = os.path.join(PWD, "one_step_test_logs")
TIMESTEP = "20160801.001500"


@pytest.fixture
def timestep_dir():

    with tempfile.TemporaryDirectory() as tmpdir:
        tstep_dir = os.path.join(tmpdir, TIMESTEP)
        os.makedirs(tstep_dir)

        yield tstep_dir


@pytest.fixture
def run_dir(timestep_dir):

    yield str(pathlib.Path(timestep_dir).parent)


def test_runs_complete_no_logfile(run_dir):

    # no logfile copied in so should be empty
    complete_timesteps = one_step.check_runs_complete(run_dir)
    assert not complete_timesteps


@pytest.mark.parametrize(
    "logfile, expected",
    [("stdout_unfinished.log", set()), ("stdout_finished.log", set([TIMESTEP]))],
)
def test_runs_complete_with_logfile(run_dir, timestep_dir, logfile, expected):

    shutil.copy(
        os.path.join(TEST_DATA_DIR, logfile),
        os.path.join(timestep_dir, one_step.STDOUT_FILENAME),
    )

    complete_timesteps = one_step.check_runs_complete(run_dir)
    assert complete_timesteps == expected
