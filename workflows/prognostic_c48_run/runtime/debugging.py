import pdb
from mpi4py import MPI
import xarray as xr


def breakpoint(rank=0):
    if rank == MPI.COMM_WORLD.rank:
        pdb.set_trace()


def checkpoint(state, path, rank=0):
    if MPI.COMM_WORLD.rank == 0:
        variables = state.checkpoint()
        input_ = xr.Dataset({key: variables[key] for key in variables})
        input_.to_netcdf(path)
