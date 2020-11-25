import logging
import os
import f90nml
from mpi4py import MPI
from fv3gfs.util import ZarrMonitor, CubedSpherePartitioner


logger = logging.getLogger(__name__)

NML_PATH = os.environ.get("INPUT_NML_PATH")
namelist = f90nml.read(NML_PATH)
logger.info(f"Loaded namelist for ZarrMonitor from {NML_PATH}")


partitioner = CubedSpherePartitioner.from_namelist(namelist)
DUMP_PATH = os.environ.get("STATE_DUMP_PATH")
output_zarr = os.path.join(str(DUMP_PATH), "state_output.zarr")
output_monitor = ZarrMonitor(
    output_zarr,
    partitioner,
    mpi_comm=MPI.COMM_WORLD
)
logger.info(f"Initialized zarr monitor at: {output_zarr}")


def print_rank(state):
    logger.info(MPI.COMM_WORLD.Get_rank())
