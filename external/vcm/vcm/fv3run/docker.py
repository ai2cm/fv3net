import os
import sys
from os.path import abspath, basename, join
from shutil import copy, copytree, rmtree
from subprocess import check_call

from vcm import casting
from vcm.cubedsphere import save_tiles_separately

from .common import write_submit_job_script

DOCKER_IMAGE = os.environ.get("FV3NET_IMAGE", "us.gcr.io/vcm-ml/fv3net")


def copy_files_into_directory(files, directory):
    for file in files:
        dest = join(directory, basename(file))
        copy(file, dest)


def add_if_present(path, destination):
    if path is not None:
        copy(path, destination)


def make_experiment(
    dir, args, template_dir="", oro_paths=(), vertical_grid=None, files_to_copy=()
):
    rmtree(dir)
    rundir = f"{dir}/rundir"
    input_dir = f"{dir}/rundir/INPUT"
    copytree(template_dir, dir)

    for prefix, combined_tile_data in args:
        float_data = casting.doubles_to_floats(combined_tile_data)
        save_tiles_separately(float_data, prefix, output_directory=input_dir)

    copy_files_into_directory(oro_paths, input_dir)

    for (src, dst) in files_to_copy:
        add_if_present(src, join(rundir, dst))

    add_if_present(vertical_grid, join(input_dir, "fv_core.res.nc"))
    return dir


def rundir(directory):
    return join(abspath(directory))


def run_experiment(directory):
    script_path = write_submit_job_script(rundir(directory))
    print("Using Docker image", DOCKER_IMAGE)
    return check_call(
        [
            "docker",
            "run",
            "-v",
            rundir(directory) + ":" + rundir(directory),
            "-v",
            "/home:/home",
            #'-v', os.getcwd() + '/data/inputdata/fv3gfs-data-docker/fix.v201702:/inputdata/fix.v201702',
            DOCKER_IMAGE,
            "bash",
            script_path,
        ]
    )


def swap_sfc_data(sfc_data, broken_sfc_data, key):
    return sfc_data.assign({key: broken_sfc_data[key]})


if __name__ == "__main__":
    run_experiment(sys.argv[1])
