# flake8: noqa
# Tensorflow looks at sys args which are not initialized
# when this module is loaded under callpyfort, so ensure
# it's available here
import sys
import os
from mpi4py import MPI

if not hasattr(sys, "argv"):
    sys.argv = [""]

# Mechanism to make only 1 GPU visible to each process
# Expects there to be as many GPUs as ranks here #TODO make this general
rank = MPI.COMM_WORLD.Get_rank()
os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

from emulation.config import (
    get_hooks,
    EmulationConfig,
    ModelConfig,
    StorageConfig,
)


gscond, microphysics, store = get_hooks()
