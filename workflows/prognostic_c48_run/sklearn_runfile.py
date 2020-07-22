import logging
from typing import MutableMapping, Hashable, cast

import fsspec
import zarr
from sklearn.externals import joblib
import xarray as xr

import fv3gfs
import runtime
from fv3net.regression.sklearn.adapters import RenamingAdapter, StackingAdapter

from mpi4py import MPI

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

State = MutableMapping[Hashable, xr.DataArray]

# following variables are required no matter what feature set is being used
TEMP = "air_temperature"
SPHUM = "specific_humidity"
DELP = "pressure_thickness_of_atmospheric_layer"
PRECIP_RATE = "surface_precipitation_rate"
TOTAL_PRECIP = "total_precipitation"  # has units of m
REQUIRED_VARIABLES = {TEMP, SPHUM, DELP, PRECIP_RATE, TOTAL_PRECIP}

cp = 1004
gravity = 9.81
m_per_mm = 1 / 1000


def compute_diagnostics(state, diags):

    net_moistening = (diags["dQ2"] * state[DELP] / gravity).sum("z")
    physics_precip = state[PRECIP_RATE]

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
        physics_precip=(physics_precip)
        .assign_attrs(units="kg/m^2/s")
        .assign_attrs(
            description="surface precipitation rate due to parameterized physics"
        ),
    )


def rename_diagnostics(diags):
    """Postfix ML output names with _diagnostic and create zero-valued outputs in
    their stead. Function operates in place."""
    for variable in ["net_moistening", "net_heating"]:
        attrs = diags[variable].attrs
        diags[f"{variable}_diagnostic"] = diags[variable].assign_attrs(
            description=attrs["description"] + " (diagnostic only)"
        )
        diags[variable] = xr.zeros_like(diags[variable]).assign_attrs(attrs)


def precipitation_sum(
    physics_precip: xr.DataArray, column_dq2: xr.DataArray, dt: float
) -> xr.DataArray:
    """Return sum of physics precipitation and ML-induced precipitation. Output is
    thresholded to enforce positive precipitation.

    Args:
        physics_precip: precipitation from physics parameterizations [m]
        column_dq2: column-integrated moistening from ML [kg/m^2/s]
        dt: physics timestep [s]

    Returns:
        total precipitation [m]"""
    ml_precip = -column_dq2 * dt * m_per_mm  # type: ignore
    total_precip = physics_precip + ml_precip
    total_precip = total_precip.where(total_precip >= 0, 0)
    total_precip.attrs["units"] = "m"
    return total_precip


def open_model(config):
    # Load the model
    rename_in = config.get("input_standard_names", {})
    rename_out = config.get("output_standard_names", {})
    with fsspec.open(config["model"], "rb") as f:
        model = joblib.load(f)

    stacked_predictor = StackingAdapter(model, sample_dims=["y", "x"])
    return RenamingAdapter(stacked_predictor, rename_in, rename_out)


def predict(model: RenamingAdapter, state: State) -> State:
    """Given ML model and state, return tendency prediction."""
    ds = xr.Dataset(state)  # type: ignore
    output = model.predict(ds)
    return {key: cast(xr.DataArray, output[key]) for key in output.data_vars}


def apply(state: State, tendency: State, dt: float) -> State:
    """Given state and tendency prediction, return updated state.
    Returned state only includes variables updated by ML model."""
    with xr.set_options(keep_attrs=True):
        updated = {
            SPHUM: state[SPHUM] + tendency["dQ2"] * dt,
            TEMP: state[TEMP] + tendency["dQ1"] * dt,
        }
    return updated  # type: ignore


class TimeLoop:
    def __init__(self, comm=None, _fv3gfs=None):

        if _fv3gfs is None:
            self.fv3gfs = fv3gfs
        else:
            self.fv3gfs = _fv3gfs

        if comm is None:
            comm = MPI.COMM_WORLD

        args = runtime.get_config()
        NML = runtime.get_namelist()
        TIMESTEP = NML["coupler_nml"]["dt_atmos"]

        rank = comm.Get_rank()

        self.state_mapping = runtime.DerivedFV3State(self.fv3gfs)

        # change into run directoryy
        MPI.COMM_WORLD.barrier()  # wait for master rank to write run directory

        self.do_only_diagnostic_ml = args["scikit_learn"].get("diagnostic_ml", False)

        # open zarr tape for output

        if rank == 0:
            logger.info("Downloading Sklearn Model")
            MODEL = open_model(args["scikit_learn"])
            logger.info("Model downloaded")
        else:
            MODEL = None

        MODEL = comm.bcast(MODEL, root=0)

        if rank == 0:
            logger.info(f"Timestep: {TIMESTEP}")

        self.rank = rank
        self.comm = comm
        self.model = MODEL
        self.timestep = TIMESTEP
        self.diagnostics = {}

    def step_dynamics(self):
        if self.rank == 0:
            logger.debug(f"Dynamics Step")
        self.fv3gfs.step_dynamics()

    def reset_diagnostics(self):
        self.diagnostics = {}

    def step_physics(self):
        if self.rank == 0:
            logger.debug(f"Physics Step")
        self.fv3gfs.step_physics()

    def step_python(self, TIMESTEP):
        variables = list(self.model.input_vars_ | REQUIRED_VARIABLES)
        if self.rank == 0:
            logger.debug(f"Getting state variables: {variables}")
        state = {name: self.state_mapping[name] for name in variables}

        if self.rank == 0:
            logger.debug("Computing RF updated variables")
        tendency = predict(self.model, state)

        if self.do_only_diagnostic_ml:
            updated_state: State = {}
        else:
            updated_state = apply(state, tendency, dt=TIMESTEP)

        diagnostics = compute_diagnostics(state, tendency)
        if self.do_only_diagnostic_ml:
            rename_diagnostics(diagnostics)

        updated_state[TOTAL_PRECIP] = precipitation_sum(
            state[TOTAL_PRECIP], diagnostics["net_moistening"], TIMESTEP
        )

        if self.rank == 0:
            logger.debug("Setting Fortran State")

        self.state_mapping.update(updated_state)
        self.diagnostics.update(diagnostics)

    def step(self, TIMESTEP):
        self.reset_diagnostics()
        self.step_dynamics()
        self.step_physics()
        self.step_python(TIMESTEP)
        return self.state_mapping.time, self.diagnostics

    def __iter__(self):
        self.fv3gfs.initialize()
        for i in range(self.fv3gfs.get_step_count()):
            yield self.step(self.timestep)
        self.fv3gfs.cleanup()


if __name__ == "__main__":
    comm = MPI.COMM_WORLD

    if comm.rank == 0:
        group = zarr.open_group(
            runtime.get_config()["scikit_learn"]["zarr_output"], mode="w"
        )
    else:
        group = None

    group = comm.bcast(group, root=0)

    for i, (_, diagnostics) in enumerate(TimeLoop(comm=comm)):
        if i == 0:
            writers = runtime.init_writers(group, comm, diagnostics)
        runtime.append_to_writers(writers, diagnostics)
