import cftime
import json
import logging
import copy
from typing import (
    Any,
    Hashable,
    Iterable,
    Mapping,
    MutableMapping,
    Tuple,
    cast,
    List,
    Sequence,
)

import xarray as xr
from mpi4py import MPI

import fv3gfs.wrapper as wrapper

# To avoid very strange NaN errors this needs to happen before runtime import
# with openmpi
wrapper.initialize()  # noqa: E402

import fv3gfs.util
import runtime


logging.basicConfig(level=logging.DEBUG)
logging.getLogger("fv3gfs.util").setLevel(logging.WARN)
logger = logging.getLogger(__name__)

State = MutableMapping[Hashable, xr.DataArray]
Diagnostics = Mapping[str, xr.DataArray]

# following variables are required no matter what feature set is being used
TEMP = "air_temperature"
SPHUM = "specific_humidity"
DELP = "pressure_thickness_of_atmospheric_layer"
PRECIP_RATE = "surface_precipitation_rate"
TOTAL_PRECIP = "total_precipitation"  # has units of m
AREA = "area_of_grid_cell"
REQUIRED_VARIABLES = {TEMP, SPHUM, DELP, PRECIP_RATE, TOTAL_PRECIP}

cp = 1004
gravity = 9.81
m_per_mm = 1 / 1000


def global_average(comm, array, area):
    ans = comm.reduce((area * array).sum().item(), root=0)
    area_all = comm.reduce(area.sum().item(), root=0)
    if comm.rank == 0:
        return ans / area_all
    else:
        return -1


def compute_diagnostics(state, diags):

    net_moistening = (diags["dQ2"] * state[DELP] / gravity).sum("z")
    physics_precip = state[PRECIP_RATE]

    return dict(
        air_temperature=state[TEMP],
        specific_humidity=state[SPHUM],
        pressure_thickness_of_atmospheric_layer=state[DELP],
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
    model = runtime.get_ml_model(config)
    rename_in = config.get("input_standard_names", {})
    rename_out = config.get("output_standard_names", {})
    return runtime.RenamingAdapter(model, rename_in, rename_out)


def predict(model: runtime.RenamingAdapter, state: State, true_tends) -> State:
    """Given ML model and state, return tendency prediction."""
    ds = xr.Dataset(state)  # type: ignore
    output = model.predict_columnwise(ds, feature_dim="z")
    # a = xr.Dataset(true_tends)
    # threshold_tendency = 0.01 / 86400  # K/day
    # active = (
    #     (a.dQ1 ** 2 + (2.51e6 / 1004 * a.dQ2) ** 2) > threshold_tendency ** 2
    # ).any("z")
    # output = output.where(active, 0.0)
    # percent_active = (active.sum() / active.count()).item()
    # logging.info(json.dumps({"percent_active_ml": 100 * percent_active}))
    return {key: cast(xr.DataArray, output[key]) for key in output.data_vars}


def limit_sphum_tendency(state: State, tendency: State, dt: float):
    delta = tendency["dQ2"] * dt
    tendency_updated = copy.copy(tendency)
    tendency_updated["dQ2"] = xr.where(
        state[SPHUM] + delta > 0, tendency["dQ2"], -state[SPHUM] / dt,  # type: ignore
    )
    log_updated_tendencies(tendency, tendency_updated)
    return tendency_updated


def log_updated_tendencies(tendency: State, tendency_updated: State):
    rank_updated_points = xr.where(tendency["dQ2"] != tendency_updated["dQ2"], 1, 0)
    updated_points = comm.reduce(rank_updated_points, root=0)
    if comm.rank == 0:
        level_updates = {
            i: int(value)
            for i, value in enumerate(updated_points.sum(["x", "y"]).values)
        }
        logging.info(f"specific_humidity_limiter_updates_per_level: {level_updates}")


def apply(state: State, tendency: State, dt: float) -> State:
    """Given state and tendency prediction, return updated state.
    Returned state only includes variables updated by ML model."""

    with xr.set_options(keep_attrs=True):
        updated = {
            SPHUM: state[SPHUM] + tendency["dQ2"] * dt,
            TEMP: state[TEMP] + tendency["dQ1"] * dt,
        }
    return updated  # type: ignore


class TimeLoop(Iterable[Tuple[cftime.DatetimeJulian, Diagnostics]]):
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

    def __init__(self, comm: Any = None, fv3gfs=wrapper) -> None:

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

        # download the model
        self._log_info("Downloading ML Model")
        self._model = open_model(args["scikit_learn"])
        self._log_info("Model Downloaded")
        MPI.COMM_WORLD.barrier()  # wait for initialization to finish

    @property
    def time(self) -> cftime.DatetimeJulian:
        return self._state.time

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

        micro = fv3gfs.wrapper.get_diagnostic_by_name(
            "tendency_of_specific_humidity_due_to_microphysics"
        ).data_array
        delp = self._state[DELP]
        return {
            "micro": (micro * delp).sum("z") / 9.81,
            "evap": fv3gfs.wrapper.get_diagnostic_by_name("lhtfl").data_array / 2.51e6,
            "cnvprcp_after_physics": fv3gfs.wrapper.get_diagnostic_by_name(
                "cnvprcp"
            ).data_array,
        }

    def _step_python(self) -> Diagnostics:

        variables: List[Hashable] = list(
            self._model.input_variables | REQUIRED_VARIABLES
        )
        self._log_debug(f"Getting state variables: {variables}")
        state = {name: self._state[name] for name in variables}

        import fv3gfs.wrapper

        # self._log_debug("Computing RF updated variables")
        true_tends = dict(
            dQ1=fv3gfs.wrapper.get_diagnostic_by_name(
                "tendency_of_air_temperature_due_to_deep_convection"
            ).data_array,
            dQ2=fv3gfs.wrapper.get_diagnostic_by_name(
                "tendency_of_specific_humidity_due_to_deep_convection"
            ).data_array,
        )
        tendency = predict(self._model, state, true_tends)

        self._log_debug("Getting physics tendencies from physics diags")
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

        return {
            "delp": self._state[DELP],
            "area": self._state[AREA],
            "cnvprcp_after_python": fv3gfs.wrapper.get_diagnostic_by_name(
                "cnvprcp"
            ).data_array,
            **diagnostics,
        }

    def __iter__(self):
        for i in range(self._fv3gfs.get_step_count()):
            diagnostics = {}
            diagnostics.update(self._step_dynamics())
            diagnostics.update(self._step_physics())
            diagnostics.update(self._step_python())
            yield self._state.time, diagnostics
        self._fv3gfs.cleanup()


def monitor(name: str, func):
    """Decorator to add tendency monitoring to an update function

    This will add `tendency_of_{variable}_due_to_{name}` to the
    diagnostics and print mass conservation diagnostics to the logs.

    Args:
        name: the name to tag the tendency diagnostics with
        func: a stepping function, usually a bound method of TimeLoop

    Returns:
        monitored function. Same as func, but with tendency and mass change
        diagnostics.
    """

    def step(self) -> Mapping[str, xr.DataArray]:
        area = self._state[AREA]
        before = {key: self._state[key] for key in self._variables + [DELP]}
        mass_before = comm.reduce((area * before[DELP]).sum().item(), root=0)
        vapor_mass_before = comm.reduce(
            (area * before[DELP] * self._state[SPHUM]).sum().item(), root=0
        )
        diags = func(self)
        after = {key: self._state[key] for key in self._variables + [DELP]}
        mass_after = comm.reduce((area * after[DELP]).sum().item(), root=0)
        vapor_mass_after = comm.reduce(
            (area * after[DELP] * self._state[SPHUM]).sum().item(), root=0
        )
        area_all = comm.reduce(area.sum().item(), root=0)

        tendency = {
            f"tendency_of_{key}_due_to_{name}": (after[key] - before[key])
            / self._timestep
            for key in self._variables
        }

        if comm.rank == 0:
            output = {
                "total_mass_change": {
                    "value": (mass_after - mass_before) / area_all,
                    "units": "Pa",
                },
                "vapor_mass_change": {
                    "value": (vapor_mass_after - vapor_mass_before) / area_all,
                    "units": "Pa",
                },
            }

            logging.info(f"{name}:{json.dumps(output)}")

        return {**diags, **tendency}

    return step


class MonitoredPhysicsTimeLoop(TimeLoop):
    def __init__(self, tendency_variables: Sequence[str], *args, **kwargs):
        """

        Args:
            tendency_variables: a list of variables to compute the physics
                tendencies of.
                
        """
        super().__init__(*args, **kwargs)
        self._variables = list(tendency_variables)

    _step_physics = monitor("fv3_physics", TimeLoop._step_physics)
    _step_python = monitor("python", TimeLoop._step_python)


if __name__ == "__main__":
    comm = MPI.COMM_WORLD

    config = runtime.get_config()
    partitioner = fv3gfs.util.CubedSpherePartitioner.from_namelist(config["namelist"])

    loop = MonitoredPhysicsTimeLoop(
        tendency_variables=config.get("scikit_learn", {}).get(
            "physics_tendency_vars", []
        ),
        comm=comm,
    )

    diag_files = runtime.get_diagnostic_files(
        config, partitioner, comm, initial_time=loop.time
    )

    for time, diagnostics in loop:

        averages = {}
        for v in [
            "net_moistening",
            "evap",
            "micro",
            "cnvprcp_after_python",
            "cnvprcp_after_physics",
        ]:
            averages[v] = (
                global_average(comm, diagnostics[v], diagnostics["area"]) * 86400
            )

        if comm.rank == 0:
            logging.info(json.dumps(averages))

        for diag_file in diag_files:
            diag_file.observe(time, diagnostics)
