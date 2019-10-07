import os
import sys
from subprocess import call
from os.path import abspath, join
from shutil import copytree, copy


def make_experiment(name, sfc_data, namelist_path, template_dir):
    dir=f"experiments/{name}"
    if os.path.exists(dir):
        return dir
    rundir=f"{dir}/rundir"
    input_dir=f"{dir}/rundir/INPUT"
    copytree(template_dir, dir)
    save_surface_data(sfc_data, output_directory=input_dir)
    copy(namelist_path, join(rundir, 'input.nml'))

    return dir


def save_surface_data(sfc_data, output_directory):
    for i in range(6):
        output_path = join(output_directory, f"sfc_data.tile{i+1}.nc")
        sfc_data.isel(tiles=i).to_netcdf(output_path)


def rundir(directory):
    return join(abspath(directory), 'rundir')


def run_experiment(directory):
    return call([
        'docker', 'run', '-d',
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
