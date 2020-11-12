"""
A module for testing/debugging call routines
"""
import logging
import numpy as np


def print_arr_info(state):

    logger = logging.getLogger(__name__)
    for varname, arr in state.items():
        logger.info(f"{varname}: shape[{arr.shape}] isfortran[{np.isfortran(arr)}]")