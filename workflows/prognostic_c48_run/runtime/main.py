import gc
import logging
from mpi4py import MPI

import fv3gfs.wrapper as wrapper

# To avoid very strange NaN errors this needs to happen before runtime import
# with openmpi
wrapper.initialize()  # noqa: E402

import tensorflow as tf
from runtime.loop import TimeLoop
import pace.util as util
import runtime

STATISTICS_LOG_NAME = "statistics"
PROFILES_LOG_NAME = "profiles"

logging.basicConfig(level=logging.INFO)

logging.getLogger("pace.util").setLevel(logging.WARN)
logging.getLogger("fsspec").setLevel(logging.WARN)
logging.getLogger("urllib3").setLevel(logging.WARN)

# Fortran logs are output as python DEBUG level
runtime.capture_fv3gfs_funcs()

logger = logging.getLogger(__name__)


def limit_visible_gpus_by_rank():
    """
    Limits the GPUs available to a given rank to spread out work
    """
    rank = MPI.COMM_WORLD.Get_rank()
    physical_devices = tf.config.list_physical_devices("GPU")
    num_gpus = len(physical_devices)

    # TODO: maybe handle rank < num_gpus?
    gpu_index = rank % num_gpus
    use_devices = [physical_devices[gpu_index]]
    tf.config.experimental.set_visible_devices(use_devices, "GPU")
    logical_gpus = tf.config.list_logical_devices("GPU")
    logger.info(f"Available gpu list on rank {rank}: {logical_gpus}")


def disable_tensorflow_gpu_preallocation():
    """
    Enables "memory growth" option on all gpus for tensorflow.

    Without this, tensorflow will eagerly allocate all gpu memory,
    leaving none for pytorch.
    """
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info("set memory growth for %d physical gpus")


def main():
    comm = MPI.COMM_WORLD

    disable_tensorflow_gpu_preallocation()
    limit_visible_gpus_by_rank()
    config = runtime.get_config()
    partitioner = util.CubedSpherePartitioner.from_namelist(runtime.get_namelist())
    for name in [STATISTICS_LOG_NAME, PROFILES_LOG_NAME]:
        runtime.setup_file_logger(name)

    loop = TimeLoop(config, comm=comm)

    diag_files = runtime.get_diagnostic_files(
        config.diagnostics, partitioner, comm, initial_time=loop.time
    )
    if comm.rank == 0:
        runtime.write_chunks(config)

    writer = tf.summary.create_file_writer(f"tensorboard/rank_{comm.rank}")

    with writer.as_default():
        for time, diagnostics in loop:

            if comm.rank == 0:
                logger.debug(f"diags: {list(diagnostics.keys())}")

            averages = runtime.globally_average_2d_diagnostics(
                comm, diagnostics, exclude=loop._states_to_output
            )
            profiles = runtime.globally_sum_3d_diagnostics(
                comm, diagnostics, ["specific_humidity_limiter_active"]
            )
            if comm.rank == 0:
                runtime.log_mapping(time, averages, STATISTICS_LOG_NAME)
                runtime.log_mapping(time, profiles, PROFILES_LOG_NAME)

            for diag_file in diag_files:
                diag_file.observe(time, diagnostics)

    # Diag files *should* flush themselves on deletion but
    # fv3gfs.wrapper.cleanup short-circuits the usual python deletion
    # mechanisms
    for diag_file in diag_files:
        diag_file.flush()

    loop.log_global_timings()


if __name__ == "__main__":

    main()
    # need to cleanup any python objects that may have MPI operations before
    # calling wrapper.cleanup
    # this avoids the following error message:
    #
    #    Attempting to use an MPI routine after finalizing MPICH
    gc.collect()
    wrapper.cleanup()
