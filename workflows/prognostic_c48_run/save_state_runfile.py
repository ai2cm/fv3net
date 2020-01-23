import os

import fsspec

import fv3gfs
import state_io
from mpi4py import MPI


def get_names(props):
    for var in props:
        yield var["name"]


VARIABLES = list(state_io.CF_TO_RESTART_MAP)


rundir_basename = "rundir"
output_path = "/code/state.pkl"

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    current_dir = os.getcwd()
    rundir_path = os.path.join(current_dir, rundir_basename)
    MPI.COMM_WORLD.barrier()  # wait for master rank to write run directory
    os.chdir(rundir_path)

    # Calculate factor for relaxing humidity to zero
    fv3gfs.initialize()
    fv3gfs.step_dynamics()
    fv3gfs.step_physics()
    state = fv3gfs.get_state(names=VARIABLES)
    combined = comm.gather(state, root=0)

    if rank == 0:
        with fsspec.open(output_path, "wb") as f:
            state_io.dump(combined, f)
    fv3gfs.cleanup()
