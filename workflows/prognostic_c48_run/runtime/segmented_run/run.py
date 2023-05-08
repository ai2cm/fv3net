import sys
import subprocess
import os
import glob
import contextlib
from pathlib import Path

import fv3config
from .logs import handle_fv3_log

runfile = Path(__file__).parent.parent / "main.py"


@contextlib.contextmanager
def cwd(path):
    cwd = os.getcwd()
    os.chdir(path)
    yield
    os.chdir(cwd)


def find(path: str):
    return glob.glob(os.path.join(path, "**"), recursive=True)


def compose_simulation_command(nprocs: int, mpi_launcher: str = "mpirun"):

    command = [
        mpi_launcher,
        "-n",
        str(nprocs),
        sys.executable,
        "-m",
        "mpi4py",
        runfile.absolute().as_posix(),
    ]

    if mpi_launcher == "srun":
        command.insert(1, "--export=ALL")

    if not (mpi_launcher == "mpirun" or mpi_launcher == "srun"):
        raise ValueError(
            "Unrecognized mpi_launcher option. Please choose between 'mpirun' or 'srun'"
        )

    return command


def run_segment(config: dict, rundir: str, mpi_launcher: str):
    fv3config.write_run_directory(config, rundir)
    with cwd(rundir):
        manifest = find(".")
        with open("preexisting_files.txt", "w") as f:
            for path in manifest:
                print(path, file=f)

        x, y = config["namelist"]["fv_core_nml"]["layout"]
        nprocs = x * y * 6

        command = compose_simulation_command(nprocs, mpi_launcher)

        with open("logs.txt", "w") as f:
            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            # need this assertion so that mypy knows that stdout is not None
            assert process.stdout, "stdout should not be None"
            for line in handle_fv3_log(process.stdout):
                for out_file in [sys.stdout, f]:
                    print(line.strip(), file=out_file)

        return process.wait()
