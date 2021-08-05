import sys
import subprocess
import os
import glob
import contextlib
from pathlib import Path

import fv3config

runfile = Path(__file__).parent.parent / "main.py"


@contextlib.contextmanager
def cwd(path):
    cwd = os.getcwd()
    os.chdir(path)
    yield
    os.chdir(cwd)


def find(path: str):
    return glob.glob(os.path.join(path, "**"), recursive=True)


def run_segment(config: dict, rundir: str):
    fv3config.write_run_directory(config, rundir)
    with cwd(rundir):
        manifest = find(".")
        with open("preexisting_files.txt", "w") as f:
            for path in manifest:
                print(path, file=f)

        x, y = config["namelist"]["fv_core_nml"]["layout"]
        nprocs = x * y * 6
        with open("logs.txt", "w") as f:
            subprocess.check_call(
                [
                    "mpirun",
                    "-n",
                    str(nprocs),
                    sys.executable,
                    runfile.absolute().as_posix(),
                ],
                # stdout=f,
                # stderr=f,
            )


def main():
    config_path, rundir = sys.argv[1:]
    with open(config_path) as f:
        config = fv3config.load(f)
    run_segment(config, rundir)
