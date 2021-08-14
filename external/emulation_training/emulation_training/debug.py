"""
A module for testing/debugging call routines
"""
import logging
import os
import traceback
import numpy as np
from mpi4py import MPI
from datetime import datetime

logger = logging.getLogger(__name__)


def print_errors(func):
    def new_func(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(traceback.print_exc())
            raise e

    return new_func


def print_arr_info(state):

    logger = logging.getLogger(__name__)
    for varname, arr in state.items():
        logger.info(f"{varname}: shape[{arr.shape}] isfortran[{np.isfortran(arr)}]")


def print_location_ping(state):

    logger = logging.getLogger(__name__)
    logger.info("Ping reached!")


def dump_state(state):

    DUMP_PATH = os.path.join(os.getcwd(), "debug_dump")
    os.makedirs(DUMP_PATH, exist_ok=True)

    logger = logging.getLogger(__name__)
    rank = MPI.COMM_WORLD.Get_rank()

    time_str = datetime.now().strftime("%Y%m%d.%H%M%S")
    filename = f"state_dump.{time_str}.tile{rank}.npz"
    outfile = os.path.join(DUMP_PATH, filename)
    logger.info(f"Dumping state to {outfile}")
    np.savez(outfile, **state)
