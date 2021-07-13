import sys
import subprocess
import os
import glob
import shutil
import contextlib

import fv3config


@contextlib.contextmanager
def cwd(path):
    cwd = os.getcwd()
    os.chdir(path)
    yield
    os.chdir(cwd)


def find(path: str):
    return glob.glob(os.path.join(path, "**"), recursive=True)


def run_segment(config: dict, rundir: str, runfile: str):
    fv3config.write_run_directory(config, rundir)
    shutil.copy(runfile, os.path.join(rundir, "runfile.py"))

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
                    os.path.abspath(runfile),
                ],
                stdout=f,
                stderr=f,
            )


def main():
    config_path, rundir, runfile = sys.argv[1:]
    with open(config_path) as f:
        config = fv3config.load(f)
    run_segment(config, rundir, runfile)
