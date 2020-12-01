import logging
import os
import cftime
import f90nml
import numpy as np
from mpi4py import MPI

from fv3gfs.util import ZarrMonitor, CubedSpherePartitioner, Quantity
from .debug import print_errors


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

DIMS_MAP = {
    1: ["horizontal_dimension"],
    2: ["lev", "horizontal_dimension"],
    3: ["tracer_number", "lev", "horizontal_dimension"]
}


def print_rank(state):
    logger.info(MPI.COMM_WORLD.Get_rank())


def _adjust_vert_plus_one(data, dims):
    logger.debug("Found vertical layers plus 1, adjusting dim name")
    dim_idx = dims.index("lev")
    dims = list(dims)
    if data.shape[dim_idx] == 80:
        dims[dim_idx] = "lev_plus_one"
    return dims


def _convert_to_quantites(state):

    quantities = {}
    for key, data in state.items():
        data = np.squeeze(data.astype(np.float32))
        dims = DIMS_MAP[data.ndim]
        if "lev" in dims:
            dims = _adjust_vert_plus_one(data, dims)
        quantities[key] = Quantity(data, dims, "units")

    return quantities


def translate_time(time):
    year = time[0]
    month = time[1]
    day = time[2]
    hour = time[4]
    min = time[5]
    datetime = cftime.DatetimeJulian(year, month, day, hour, min)
    logger.debug(f"Translated time: {datetime}")

    return datetime


@print_errors
def store(state):
    logger.info(f"Storing model state on rank {MPI.COMM_WORLD.Get_rank()}")
    logger.debug(f"Model fields: {list(state.keys())}")
    time = translate_time(state.pop("model_time"))
    state = _convert_to_quantites(state)
    state["time"] = time
    output_monitor.store(state)
