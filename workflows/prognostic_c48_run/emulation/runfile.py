import logging
import fsspec
import functools
import yaml
import numpy as np
from datetime import timedelta

import fv3gfs.util


if __name__ == "__main__":
    import fv3gfs.wrapper as wrapper
    from mpi4py import MPI
else:
    wrapper = None
    MPI = None

#layer phi - could just sum delz
#layer p - could just sum delp

STORE_NAMES = [
    "time",
    "latitude",
    "longitude",
    "mean_cos_zenith_angle",
    "eastward_wind",
    "northward_wind",
    "vertical_wind",
    "air_temperature",
    "specific_humidity",
    
]


def get_timestep(config):
    return timedelta(seconds=config["namelist"]["coupler_nml"]["dt_atmos"])


def master_only(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        if MPI.COMM_WORLD.Get_rank() == 0:
            result = func(*args, **kwargs)
        else:
            result = None
        return MPI.COMM_WORLD.bcast(result)

    return wrapped


@master_only
def load_config(filename):
    with open("fv3config.yml", "r") as config_file:
        config = yaml.safe_load(config_file)
    return config


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config = load_config("fv3config.yml")
    partitioner = fv3gfs.util.CubedSpherePartitioner.from_namelist(config["namelist"])
    communicator = fv3gfs.util.CubedSphereCommunicator(MPI.COMM_WORLD, partitioner)
    timestep = get_timestep(config)

    store = fsspec.get_mapper("file://mnt/disks/scratch/all_physics_emu/state.zarr")
    monitor = fv3gfs.util.ZarrMonitor(store, partitioner, mpi_comm=communicator)

    wrapper.initialize()
    for i in range(wrapper.get_step_count()):
        state = wrapper.get_state(names=STORE_NAMES)
        state["time"] += timestep  # consistent with Fortran diagnostic output times
        time = state["time"]
        wrapper.step_dynamics()
        wrapper.step_physics()
        monitor.store(state)

    wrapper.cleanup()
