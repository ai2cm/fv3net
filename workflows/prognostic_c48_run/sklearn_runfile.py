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

# following variables are required no matter what feature set is being used
TEMP = "air_temperature"
SPHUM = "specific_humidity"
DELP = "pressure_thickness_of_atmospheric_layer"
TOTAL_PRECIP = "total_precipitation"
REQUIRED_VARIABLES = [TEMP, SPHUM, DELP, TOTAL_PRECIP]

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

    rename_ML_to_CF = args["scikit_learn"].get("input_variable_standard_names", {})
    rename_CF_to_ML = dict(zip(rename_ML_to_CF.values(), rename_ML_to_CF.keys()))
    if rank == 0:
        logger.debug(f"Renaming variables for ML prediction using: {rename_CF_to_ML}")

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
    variables = list(set(REQUIRED_VARIABLES + MODEL.input_vars_))
    variables = [rename_ML_to_CF.get(var, var) for var in variables]
    if rank == 0:
        logger.debug(f"Prognostic run requires variables: {variables}")

    if rank == 0:
        logger.info(f"Timestep: {TIMESTEP}")

    fv3gfs.initialize()
    for i in range(fv3gfs.get_step_count()):
        if rank == 0:
            logger.debug(f"Dynamics Step")
        fv3gfs.step_dynamics()
        if rank == 0:
            logger.debug(f"Physics Step")
        fv3gfs.step_physics()

        if rank == 0:
            logger.debug(f"Getting state variables: {variables}")
        state = {
            key: value.data_array
            for key, value in fv3gfs.get_state(names=variables).items()
        }

        if rank == 0:
            logger.debug("Computing RF updated variables")
        preds, diags = update(
            MODEL, runtime.rename_keys(state, rename_CF_to_ML), dt=TIMESTEP
        )

        if rank == 0:
            logger.debug("Setting Fortran State")
        preds = runtime.rename_keys(preds, rename_ML_to_CF)
        fv3gfs.set_state(
            {key: fv3util.Quantity.from_data_array(preds[key]) for key in preds}
        )

        diagnostics = compute_diagnostics(state, diags)

        if i == 0:
            writers = runtime.init_writers(GROUP, comm, diagnostics)
        runtime.append_to_writers(writers, diagnostics)

        times.append(get_time())

    fv3gfs.cleanup()
