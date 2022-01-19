import logging
from mpi4py import MPI


def setup_logging():
    logger = logging.getLogger("emulation")
    if MPI.COMM_WORLD.rank != 0:
        logger.setLevel(logging.CRITICAL)
    else:
        logger.setLevel(logging.DEBUG)
