import sys
import subprocess
import fv3config
import os
import glob


def find(path: str):
    return glob.glob(os.path.join(path, "**"), recursive=True)


def run_segment(config: dict, rundir: str, runfile: str):
    fv3config.write_run_directory(config, rundir)

    manifest = find(rundir)
    with open("preexisting-files.txt", "w") as f:
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
                "-m",
                os.path.abspath(runfile),
            ],
            stdout=f,
            stderr=f,
        )
