import logging
import os

import zarr

import f90nml
import fv3gfs
import run_sklearn
import state_io
from fv3gfs._wrapper import get_time
from mpi4py import MPI

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

SPHUM = "specific_humidity"
DELP = "pressure_thickness_of_atmospheric_layer"
VARIABLES = list(state_io.CF_TO_RESTART_MAP) + [DELP]

cp = 1004
gravity = 9.81


def compute_diagnostics(state, diags):
    return dict(
        net_precip=(diags["Q2"] * state[DELP] / 9.81)
        .sum("z")
        .assign_attrs(units="kg/m^2/s"),
        PW=(state[SPHUM] * state[DELP] / gravity).sum("z").assign_attrs(units="mm"),
        net_heating=(diags["Q1"] * state[DELP] / gravity * cp)
        .sum("z")
        .assign_attrs(units="W/m^2"),
    )


def init_writers(group, comm, diags):
    return {key: state_io.ZarrVariableWriter(comm, group, name=key) for key in diags}


def append_to_writers(writers, diags):
    for key in writers:
        writers[key].append(diags[key])


rundir_basename = "rundir"
input_nml = "rundir/input.nml"
NML = f90nml.read(input_nml)
TIMESTEP = NML["coupler_nml"]["dt_atmos"]

times = []

if __name__ == "__main__":
    comm = MPI.COMM_WORLD

    group = zarr.open_group("test.zarr", mode="w")

    rank = comm.Get_rank()

    if rank == 0:
        logger.info("Downloading Sklearn Model")
        MODEL = run_sklearn.open_sklearn_model(run_sklearn.SKLEARN_MODEL)
        logger.info("Model downloaded")
    else:
        MODEL = None

    MODEL = comm.bcast(MODEL, root=0)

    current_dir = os.getcwd()
    rundir_path = os.path.join(current_dir, rundir_basename)
    MPI.COMM_WORLD.barrier()  # wait for master rank to write run directory
    os.chdir(rundir_path)
    if rank == 0:
        logger.info(f"Timestep: {TIMESTEP}")

    # Calculate factor for relaxing humidity to zero
    fv3gfs.initialize()
    for i in range(fv3gfs.get_step_count()):
        if rank == 0:
            logger.debug(f"Dynamics Step")
        fv3gfs.step_dynamics()
        fv3gfs.step_physics()

        if rank == 0:
            logger.debug(f"Getting state variables: {VARIABLES}")
        state = fv3gfs.get_state(names=VARIABLES)

        if rank == 0:
            logger.debug("Computing RF updated variables")
        preds, diags = run_sklearn.update(MODEL, state, dt=TIMESTEP)
        if rank == 0:
            logger.debug("Setting Fortran State")
        fv3gfs.set_state(preds)
        if rank == 0:
            logger.debug("Setting Fortran State")

        diagnostics = compute_diagnostics(state, diags)

        if i == 0:
            writers = init_writers(group, comm, diagnostics)
        append_to_writers(writers, diagnostics)

        times.append(get_time())

    fv3gfs.cleanup()
