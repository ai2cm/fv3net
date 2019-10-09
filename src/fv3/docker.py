import os
import sys
from subprocess import call
from os.path import abspath, join, basename
import os
from shutil import copytree, copy, rmtree
import logging
from src import utils


def copy_files_into_directory(files, directory):
    for file in files:
        dest = join(directory, basename(file))
        copy(file, dest)


def make_experiment(dir,  args, namelist_path='', template_dir='', oro_paths=(),
                    vertical_grid=None):
    rmtree(dir)
    rundir=f"{dir}/rundir"
    input_dir=f"{dir}/rundir/INPUT"
    copytree(template_dir, dir)

    for prefix, combined_tile_data in args:
        float_data = utils.cast_doubles_to_floats(combined_tile_data)
        save_tiles_separately(
            float_data, prefix, output_directory=input_dir)

    copy(namelist_path, join(rundir, 'input.nml'))
    copy_files_into_directory(oro_paths, input_dir)

    if vertical_grid is not None:
        copy(vertical_grid, join(rundir, 'fv_core.res.nc'))

    return dir


def save_tiles_separately(sfc_data, prefix, output_directory):
    #TODO: move to src.data.cubedsphere
    for i in range(6):
        output_path = join(output_directory, f"{prefix}.tile{i+1}.nc")
        logging.info(f"saving data to {output_path}")
        sfc_data.isel(tiles=i).to_netcdf(output_path)


def rundir(directory):
    return join(abspath(directory), 'rundir')


def run_experiment(directory):
    return call([
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
