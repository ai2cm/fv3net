"""
A module for testing/debugging call routines
"""
import logging
import os
import traceback
import numpy as np
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

    DUMP_PATH = str(os.environ.get("STATE_DUMP_PATH"))

    logger = logging.getLogger(__name__)

    try:
        rank = state.get("rank")
    except KeyError:
        logger.info("Could not save state. No rank included in state.")
        return

    time_str = datetime.now().strftime("%Y%m%d.%H%M%S")
    filename = f"state_dump.{time_str}.tile{int(rank.squeeze()[0])}.npz"
    outfile = os.path.join(DUMP_PATH, filename)
    logger.info(f"Dumping state to {outfile}")
    np.savez(outfile, **state)
