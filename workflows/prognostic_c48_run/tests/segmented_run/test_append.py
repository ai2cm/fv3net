import sys
from pathlib import Path
import subprocess
from runtime.segmented_run.append import read_last_segment
from runtime.segmented_run.run import compose_simulation_command, runfile
from vcm.cloud import get_fs
import uuid
import pytest

def test_read_last_segment(tmpdir):
    date1 = "20160101.000000"
    date2 = "20160102.000000"
    arts = tmpdir.mkdir("artifacts")
    arts.mkdir(date1)
    arts.mkdir(date2)
    ans = read_last_segment(str(tmpdir))
    assert f"file://{str(tmpdir)}/artifacts/20160102.000000" == ans
    fs = get_fs(ans)
    assert fs.exists(ans)


def test_read_last_segment_gcs(tmp_path: Path):
    # This test requires network access, but so do some others
    # This tests the consistency of GCS in the sense of "reading our own writes"
    # (https://cloud.google.com/storage/docs/consistency)
    # and also any integrations with fsspec which has several layers of internal
    # caching which can break this.
    id_ = uuid.uuid4().hex
    url = "gs://vcm-ml-scratch/test_data" + "/" + id_
    last_segment = None
    for i in range(3):
        assert read_last_segment(url) == last_segment
        last_segment = url + f"/artifacts/{i}"
        dest = last_segment + "/hello.txt"
        file = tmp_path / "hello.txt"
        file.write_text("hello")
        subprocess.check_call(["gsutil", "cp", file.as_posix(), dest])

@pytest.mark.parametrize("mpi_launcher", ["srun", "mpirun", None, "doesnotexistrun"])
def test_compose_simulation_command(mpi_launcher):

    nprocs = "10"
    runfile_as_str = runfile.absolute().as_posix()
    sys_exe = sys.executable

    if mpi_launcher == "mpirun":
        expected = [mpi_launcher, '-n', str(nprocs), sys_exe, "-m", "mpi4py", runfile_as_str]
        assert expected == compose_simulation_command(nprocs, mpi_launcher)
    elif mpi_launcher == "srun":
        expected = [mpi_launcher, '--export=ALL', '-n', str(nprocs), sys_exe, "-m", "mpi4py", runfile_as_str]
        assert expected == compose_simulation_command(nprocs, mpi_launcher)
    elif mpi_launcher is None:
        expected = ["mpirun", '-n', str(nprocs), sys_exe, "-m", "mpi4py", runfile_as_str]
        assert expected == compose_simulation_command(nprocs)
    else:
        with pytest.raises(ValueError, match=r"Unrecognized mpi_launcher .*"):
            compose_simulation_command(nprocs, mpi_launcher)

