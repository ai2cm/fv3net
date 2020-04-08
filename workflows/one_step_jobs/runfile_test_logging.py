from mpi4py import MPI
import fv3gfs
import logging
import contextlib
import tempfile
import sys
import io
import os
import sys
import threading
import time
import logging
import sys

sys.path.insert(0, "/fv3net/workflows/one_step_jobs")
import capture


def captured_stream(func):
    def myfunc(*args, **kwargs):
        with capture.capture_stream_mpi(sys.stdout):
            return func(*args, **kwargs)

    return myfunc


def capture_fv3gfs_funcs():
    """Surpress stderr and stdout from all fv3gfs functions"""
    for func in ["step_dynamics", "step_physics", "initialize", "cleanup"]:
        setattr(fv3gfs, func, captured_stream(getattr(fv3gfs, func)))


capture_fv3gfs_funcs()

logging.basicConfig(level=logging.DEBUG)
# rank = MPI.COMM_WORLD.rank

fv3gfs.initialize()
for i in range(fv3gfs.get_step_count()):
    fv3gfs.step_dynamics()
    fv3gfs.step_physics()
    logging.info(f"timestep {i}")
    fv3gfs.save_intermediate_restart_if_enabled()
fv3gfs.cleanup()
