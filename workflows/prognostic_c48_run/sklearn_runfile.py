import logging
from mpi4py import MPI

import fv3gfs.wrapper as wrapper

# To avoid very strange NaN errors this needs to happen before runtime import
# with openmpi
wrapper.initialize()  # noqa: E402

from runtime.loop import (
    MonitoredPhysicsTimeLoop,
    globally_average_2d_diagnostics,
    setup_metrics_logger,
    log_scalar,
)
import fv3gfs.util as util
import runtime


logging.basicConfig(level=logging.DEBUG)
logging.getLogger("fv3gfs.util").setLevel(logging.WARN)
logging.getLogger("fsspec").setLevel(logging.WARN)
logging.getLogger("urllib3").setLevel(logging.WARN)

# Fortran logs are output as python DEBUG level
runtime.capture_fv3gfs_funcs()

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    comm = MPI.COMM_WORLD

    config = runtime.get_config()
    partitioner = util.CubedSpherePartitioner.from_namelist(runtime.get_namelist())
    setup_metrics_logger()

    loop = MonitoredPhysicsTimeLoop(
        config=config,
        comm=comm,
        tendency_variables=config.step_tendency_variables,
        storage_variables=config.step_storage_variables,
    )

    diag_files = runtime.get_diagnostic_files(
        config.diagnostics, partitioner, comm, initial_time=loop.time
    )
    if comm.rank == 0:
        runtime.write_chunks(config)

    for time, diagnostics in loop:

        if comm.rank == 0:
            logger.info(f"diags: {list(diagnostics.keys())}")

        averages = globally_average_2d_diagnostics(
            comm, diagnostics, exclude=loop._states_to_output
        )
        if comm.rank == 0:
            log_scalar(time, averages)

        for diag_file in diag_files:
            diag_file.observe(time, diagnostics)

    # Diag files *should* flush themselves on deletion but
    # fv3gfs.wrapper.cleanup short-circuits the usual python deletion
    # mechanisms
    for diag_file in diag_files:
        diag_file.flush()

    loop.cleanup()
