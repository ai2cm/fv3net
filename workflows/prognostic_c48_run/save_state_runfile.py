import fsspec

import fv3gfs
import state_io
from mpi4py import MPI


def get_names(props):
    for var in props:
        yield var["name"]


VARIABLES = list(state_io.CF_TO_RESTART_MAP)


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    MPI.COMM_WORLD.barrier()  # wait for master rank to write run directory

    # Calculate factor for relaxing humidity to zero
    fv3gfs.initialize()
    fv3gfs.step_dynamics()
    fv3gfs.step_physics()
    state = fv3gfs.get_state(names=VARIABLES)
    combined = comm.gather(state, root=0)

    if rank == 0:
        with fsspec.open("state.pkl", "wb") as f:
            state_io.dump(combined, f)
    fv3gfs.cleanup()
