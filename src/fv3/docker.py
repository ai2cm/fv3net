import os
import sys
from subprocess import call, check_call
from os.path import abspath, join, basename
import os
from shutil import copytree, copy, rmtree
import logging
from src import utils


def copy_files_into_directory(files, directory):
    for file in files:
        dest = join(directory, basename(file))
        copy(file, dest)


def add_if_present(path, destination):
    if path is not None:
        copy(path, destination)


def make_experiment(dir,  args, template_dir='', oro_paths=(),
                    vertical_grid=None, files_to_copy=()):
    rmtree(dir)
    rundir=f"{dir}/rundir"
    input_dir=f"{dir}/rundir/INPUT"
    copytree(template_dir, dir)

    for prefix, combined_tile_data in args:
        float_data = utils.cast_doubles_to_floats(combined_tile_data)
        save_tiles_separately(
            float_data, prefix, output_directory=input_dir)

    copy_files_into_directory(oro_paths, input_dir)
    
    for (src, dst) in files_to_copy:
        add_if_present(src, join(rundir, dst))

    add_if_present(vertical_grid, join(input_dir, 'fv_core.res.nc'))
    return dir


def save_tiles_separately(sfc_data, prefix, output_directory):
    #TODO: move to src.data.cubedsphere
    for i in range(6):
        output_path = join(output_directory, f"{prefix}.tile{i+1}.nc")
        logging.info(f"saving data to {output_path}")
        sfc_data.isel(tile=i).to_netcdf(output_path)


def rundir(directory):
    return join(abspath(directory), 'rundir')


def run_experiment(directory):
    return check_call([
        'docker', 'run', #'-d',
        '-v', rundir(directory) + ':/FV3/rundir',
        '-v', '/home/noahb/fv3gfs/inputdata/fv3gfs-data-docker/fix.v201702:/inputdata/fix.v201702',
        'fv3gfs-compiled'
    ])


def swap_sfc_data(sfc_data, broken_sfc_data, key):
    return sfc_data.assign({key: broken_sfc_data[key]})


def rundir(directory):
    return join(abspath(directory), 'rundir')


if __name__ == '__main__':
    run_experiment(sys.argv[1])
