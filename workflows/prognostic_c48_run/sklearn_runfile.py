import logging
from typing import Mapping

import fsspec
import zarr
from sklearn.externals import joblib
from sklearn.utils import parallel_backend
import xarray as xr

import fv3gfs
from fv3gfs._wrapper import get_time
import fv3util
import runtime
from fv3net.regression.sklearn.wrapper import SklearnWrapper
from mpi4py import MPI

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

SPHUM = "specific_humidity"
DELP = "pressure_thickness_of_atmospheric_layer"
TOTAL_PRECIP = "total_precipitation"
VARIABLES = list(runtime.CF_TO_RESTART_MAP) + [DELP, TOTAL_PRECIP]

cp = 1004
gravity = 9.81


def compute_diagnostics(state, diags):

    net_moistening = (diags["dQ2"] * state[DELP] / gravity).sum("z")

    return dict(
        net_moistening=(net_moistening)
        .assign_attrs(units="kg/m^2/s")
        .assign_attrs(description="column integrated ML model moisture tendency"),
        net_heating=(diags["dQ1"] * state[DELP] / gravity * cp)
        .sum("z")
        .assign_attrs(units="W/m^2")
        .assign_attrs(description="column integrated ML model heating"),
        water_vapor_path=(state[SPHUM] * state[DELP] / gravity)
        .sum("z")
        .assign_attrs(units="mm")
        .assign_attrs(description="column integrated water vapor"),
        total_precip=(state[TOTAL_PRECIP] - net_moistening)
        .assign_attrs(units="kg/m^s/s")
        .assign_attrs(
            description="total precipitation rate at the surface (model + ML)"
        ),
    )


def open_model(url):
    # Load the model
    with fsspec.open(url, "rb") as f:
        return joblib.load(f)


def predict(model: SklearnWrapper, state: xr.Dataset) -> xr.Dataset:
    """Given ML model and state, make tendency prediction."""
    stacked = state.stack(sample=["x", "y"])
    with parallel_backend("threading", n_jobs=1):
        output = model.predict(stacked, "sample").unstack("sample")
    return output


def update(
    model: SklearnWrapper, state: Mapping[str, xr.DataArray], dt: float
) -> (Mapping[str, xr.DataArray], Mapping[str, xr.DataArray]):
    """Given ML model and state, return updated state and predicted tendencies."""
    state = xr.Dataset(state)
    tend = predict(model, state)
    with xr.set_options(keep_attrs=True):
        updated = state.assign(
            specific_humidity=state["specific_humidity"] + tend["dQ2"] * dt,
            air_temperature=state["air_temperature"] + tend["dQ1"] * dt,
        )
    return {key: updated[key] for key in updated}, {key: tend[key] for key in tend}


args = runtime.get_config()
NML = runtime.get_namelist()
TIMESTEP = NML["coupler_nml"]["dt_atmos"]

times = []

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # change into run directoryy
    MPI.COMM_WORLD.barrier()  # wait for master rank to write run directory

    # open zarr tape for output
    if rank == 0:
        GROUP = zarr.open_group(args["scikit_learn"]["zarr_output"], mode="w")
    else:
        GROUP = None

    GROUP = comm.bcast(GROUP, root=0)

    if rank == 0:
        logger.info("Downloading Sklearn Model")
        MODEL = open_model(args["scikit_learn"]["model"])
        logger.info("Model downloaded")
    else:
        MODEL = None

    MODEL = comm.bcast(MODEL, root=0)

    if rank == 0:
        logger.info(f"Timestep: {TIMESTEP}")

    # Calculate factor for relaxing humidity to zero
    fv3gfs.initialize()
    for i in range(fv3gfs.get_step_count()):
        if rank == 0:
            logger.debug(f"Dynamics Step")
        fv3gfs.step_dynamics()
        if rank == 0:
            logger.debug(f"Physics Step")
        fv3gfs.step_physics()

        if rank == 0:
            logger.debug(f"Getting state variables: {VARIABLES}")
        state = {
            key: value.data_array
            for key, value in fv3gfs.get_state(names=VARIABLES).items()
        }

        if rank == 0:
            logger.debug("Computing RF updated variables")
        preds, diags = update(MODEL, state, dt=TIMESTEP)
        if rank == 0:
            logger.debug("Setting Fortran State")
        fv3gfs.set_state(
            {key: fv3util.Quantity.from_data_array(preds[key]) for key in preds}
        )
        if rank == 0:
            logger.debug("Setting Fortran State")

        diagnostics = compute_diagnostics(state, diags)

        if i == 0:
            writers = runtime.init_writers(GROUP, comm, diagnostics)
        runtime.append_to_writers(writers, diagnostics)

        times.append(get_time())

    fv3gfs.cleanup()
