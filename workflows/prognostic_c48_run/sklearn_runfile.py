import logging
from datetime import datetime
from typing import (
    Hashable,
    Iterable,
    Mapping,
    MutableMapping,
    Tuple,
    cast,
    List,
    Sequence,
)

import fsspec
import xarray as xr
from mpi4py import MPI
from sklearn.externals import joblib

import fv3gfs
import runtime
from fv3fit.sklearn import RenamingAdapter, StackingAdapter

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

State = MutableMapping[Hashable, xr.DataArray]
Diagnostics = Mapping[str, xr.DataArray]

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
        air_temperature=state[TEMP],
        specific_humidity=state[SPHUM],
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


class TimeLoop(Iterable[Tuple[datetime, Diagnostics]]):
    """An iterable defining the master time loop of a prognostic simulation

    Yields (time, diagnostics) tuples, which can be saved using diagnostic routines.

    Note:

        Each iteration consists of three phases

        1. ``_step_dynamics``
        2. ``_step_physics``
        3. ``_step_python``

        Each phase updates the fv3gfs state and returns any computed
        diagnostics. After all these stages finish, the diagnostics they
        output are merged and yielded along with the timestep.

        These methods can be overriden to change behavior or return new
        diagnostics.
    """

    def __init__(self, comm=None, fv3gfs=fv3gfs):

        if comm is None:
            comm = MPI.COMM_WORLD

        self._fv3gfs = fv3gfs
        self._state: runtime.DerivedFV3State = runtime.DerivedFV3State(self._fv3gfs)
        self._comm = comm

        args = runtime.get_config()
        namelist = runtime.get_namelist()

        # get timestep
        timestep = namelist["coupler_nml"]["dt_atmos"]
        self._timestep = timestep
        self._log_info(f"Timestep: {timestep}")

        self._do_only_diagnostic_ml: bool = args["scikit_learn"].get(
            "diagnostic_ml", False
        )

        # download the scikit-learn model
        self._log_info("Downloading Sklearn Model")
        if comm.rank == 0:
            model = open_model(args["scikit_learn"])
        else:
            model = None
        model = comm.bcast(model, root=0)
        self._model = model
        self._log_info("Model Downloaded")
        MPI.COMM_WORLD.barrier()  # wait for initialization to finish

    def _log_debug(self, message: str):
        if self._comm.rank == 0:
            logger.debug(message)

    def _log_info(self, message: str):
        if self._comm.rank == 0:
            logger.info(message)

    def _step_dynamics(self) -> Diagnostics:
        self._log_debug(f"Dynamics Step")
        self._fv3gfs.step_dynamics()
        # no diagnostics are computed by default
        return {}

    def _step_physics(self) -> Diagnostics:
        self._log_debug(f"Physics Step")
        self._fv3gfs.step_physics()
        # no diagnostics are computed by default
        return {}

    def _step_python(self) -> Diagnostics:
        variables: List[Hashable] = list(self._model.input_vars_ | REQUIRED_VARIABLES)
        self._log_debug(f"Getting state variables: {variables}")
        state = {name: self._state[name] for name in variables}

        self._log_debug("Computing RF updated variables")
        tendency = predict(self._model, state)

        if self._do_only_diagnostic_ml:
            updated_state: State = {}
        else:
            updated_state = apply(state, tendency, dt=self._timestep)

        diagnostics = compute_diagnostics(state, tendency)
        if self._do_only_diagnostic_ml:
            rename_diagnostics(diagnostics)

        updated_state[TOTAL_PRECIP] = precipitation_sum(
            state[TOTAL_PRECIP], diagnostics["net_moistening"], self._timestep
        )

        self._log_debug("Setting Fortran State")
        self._state.update(updated_state)
        return diagnostics

    def __iter__(self):
        self._fv3gfs.initialize()
        for i in range(self._fv3gfs.get_step_count()):
            diagnostics = {}
            diagnostics.update(self._step_dynamics())
            diagnostics.update(self._step_physics())
            diagnostics.update(self._step_python())
            yield self._state.time, diagnostics
        self._fv3gfs.cleanup()


class MonitoredPhysicsTimeLoop(TimeLoop):
    def __init__(self, tendency_variables: Sequence[str], *args, **kwargs):
        """

        Args:
            tendency_variables: a list of variables to compute the physics
                tendencies of.
                
        """
        super().__init__(*args, **kwargs)
        self._variables = tendency_variables

    def _step_physics(self) -> Mapping[str, xr.DataArray]:
        before = {key: self._state[key] for key in self._variables}
        super()._step_physics()
        after = {key: self._state[key] for key in self._variables}

        tendency = {
            f"tendency_of_{key}_due_to_fv3_physics": (after[key] - before[key])
            / self._timestep
            for key in self._variables
        }
        before_physics = {key: before[key] for key in self._variables}
        return {**tendency, **before_physics}


if __name__ == "__main__":
    comm = MPI.COMM_WORLD

    config = runtime.get_config()
    partitioner = fv3gfs.CubedSpherePartitioner.from_namelist(config["namelist"])
    diag_files = runtime.get_diagnostic_files(config, partitioner, comm)

    loop = MonitoredPhysicsTimeLoop(
        tendency_variables=config.get("scikit_learn", {}).get(
            "physics_tendency_vars", []
        ),
        comm=comm,
    )

    for time, diagnostics in loop:
        for diag_file in diag_files:
            diag_file.observe(time, diagnostics)
