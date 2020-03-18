import os
from fv3net import runtime
import yaml
import xarray as xr
import numpy as np
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)


def post_process():
    logger.info("Post processing model outputs")
    CHUNK = {'time': 1, 'tile': -1}

    begin = xr.open_zarr("begin_physics.zarr")
    before = xr.open_zarr("before_physics.zarr")
    after = xr.open_zarr("after_physics.zarr")

    # make the time dims consistent
    time = begin.time
    before = before.drop('time')
    after = after.drop('time')
    begin = begin.drop('time')

    # concat data
    dt = np.timedelta64(15, 'm')
    time = np.arange(len(time)) * dt
    ds = xr.concat([begin, before, after], dim='step').assign_coords(step=['begin', 'after_dynamics', 'after_physics'], time=time)
    ds.chunk(CHUNK).to_zarr("post_processed.zarr", mode='w')


if __name__ == "__main__":
    import fv3gfs
    from mpi4py import MPI

RUN_DIR = os.path.dirname(os.path.realpath(__file__))

DELP = "pressure_thickness_of_atmospheric_layer"
TIME = 'time'
VARIABLES = list(runtime.CF_TO_RESTART_MAP) + [DELP, TIME]

rank = MPI.COMM_WORLD.Get_rank()
current_dir = os.getcwd()
config = runtime.get_config()
MPI.COMM_WORLD.barrier()  # wait for master rank to write run directory

partitioner = fv3gfs.CubedSpherePartitioner.from_namelist(config["namelist"])

before_monitor = fv3gfs.ZarrMonitor(
    os.path.join(RUN_DIR, "before_physics.zarr"),
    partitioner,
    mode="w",
    mpi_comm=MPI.COMM_WORLD,
)

after_monitor = fv3gfs.ZarrMonitor(
    os.path.join(RUN_DIR, "after_physics.zarr"),
    partitioner,
    mode="w",
    mpi_comm=MPI.COMM_WORLD,
)

begin_monitor = fv3gfs.ZarrMonitor(
    os.path.join(RUN_DIR, "begin_physics.zarr"),
    partitioner,
    mode="w",
    mpi_comm=MPI.COMM_WORLD,
)

fv3gfs.initialize()
state = fv3gfs.get_state(names=VARIABLES)
if rank == 0: logger.info("Beginning steps")
for i in range(fv3gfs.get_step_count()):
    if rank == 0: logger.info(f"step {i}")
    begin_monitor.store(state)
    fv3gfs.step_dynamics()
    state = fv3gfs.get_state(names=VARIABLES)
    before_monitor.store(state)
    fv3gfs.step_physics()
    state = fv3gfs.get_state(names=VARIABLES)
    after_monitor.store(state)

if rank == 0:
    post_process()
fv3gfs.cleanup()
