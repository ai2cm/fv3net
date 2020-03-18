import os
from fv3net import runtime
import yaml

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
for i in range(fv3gfs.get_step_count()):
    begin_monitor.store(state)
    fv3gfs.step_dynamics()
    state = fv3gfs.get_state(names=VARIABLES)
    before_monitor.store(state)
    fv3gfs.step_physics()
    state = fv3gfs.get_state(names=VARIABLES)
    after_monitor.store(state)
fv3gfs.cleanup()
