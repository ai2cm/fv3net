import cftime
import datetime
import json
import logging
import copy
import functools
from typing import (
    Any,
    Callable,
    Hashable,
    Iterable,
    Mapping,
    MutableMapping,
    Tuple,
    cast,
    List,
    Sequence,
    Optional,
)

import numpy as np
import xarray as xr
from mpi4py import MPI

import fv3gfs.wrapper as wrapper

# To avoid very strange NaN errors this needs to happen before runtime import
# with openmpi
wrapper.initialize()  # noqa: E402

from runtime import DerivedFV3State
import fv3fit
import fv3gfs.util as util
import runtime


logging.basicConfig(level=logging.DEBUG)
logging.getLogger("fv3gfs.util").setLevel(logging.WARN)
logging.getLogger("fsspec").setLevel(logging.WARN)
logging.getLogger("urllib3").setLevel(logging.WARN)
logger = logging.getLogger(__name__)
# Fortran logs are output as python DEBUG level
runtime.capture_fv3gfs_funcs()

State = MutableMapping[Hashable, xr.DataArray]
Diagnostics = MutableMapping[Hashable, xr.DataArray]

# following variables are required no matter what feature set is being used
TEMP = "air_temperature"
TOTAL_WATER = "total_water"
SPHUM = "specific_humidity"
DELP = "pressure_thickness_of_atmospheric_layer"
PRECIP_RATE = "surface_precipitation_rate"
TOTAL_PRECIP = "total_precipitation"  # has units of m
AREA = "area_of_grid_cell"
EAST_WIND = "eastward_wind_after_physics"
NORTH_WIND = "northward_wind_after_physics"
TENDENCY_TO_STATE_NAME: Mapping[Hashable, Hashable] = {
    "dQ1": TEMP,
    "dQ2": SPHUM,
    "dQu": EAST_WIND,
    "dQv": NORTH_WIND,
}
SST_NAME = "ocean_surface_temperature"
TSFC_NAME = "surface_temperature"
MASK_NAME = "land_sea_mask"

gravity = 9.81
m_per_mm = 1 / 1000


def setup_metrics_logger():
    logger = logging.getLogger("statistics")
    fh = logging.FileHandler("statistics.txt")
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(fmt="%(levelname)s:%(name)s:%(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)


def log_scalar(time, scalars):
    dt = datetime.datetime(
        time.year, time.month, time.day, time.hour, time.minute, time.second
    )
    msg = json.dumps({"time": dt.isoformat(), **scalars})
    logging.getLogger("statistics").info(msg)


def global_average(comm, array: xr.DataArray, area: xr.DataArray) -> float:
    ans = comm.reduce((area * array).sum().item(), root=0)
    area_all = comm.reduce(area.sum().item(), root=0)
    if comm.rank == 0:
        return float(ans / area_all)
    else:
        return -1


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
    model_paths = config["scikit_learn"]["model"]
    models = []
    for path in model_paths:
        model = fv3fit.load(path)
        rename_in = config.get("input_standard_names", {})
        rename_out = config.get("output_standard_names", {})
        models.append(runtime.RenamingAdapter(model, rename_in, rename_out))
    return runtime.MultiModelAdapter(models)


def predict(model: fv3fit.Predictor, state: State) -> State:
    """Given ML model and state, return tendency prediction."""
    ds = xr.Dataset(state)  # type: ignore
    output = model.predict_columnwise(ds, feature_dim="z")
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
        updated = {}
        for name in tendency:
            state_name = TENDENCY_TO_STATE_NAME.get(name, name)
            updated[state_name] = state[state_name] + tendency[name] * dt
    return updated  # type: ignore


class LoggingMixin:

    rank: int

    def _log_debug(self, message: str):
        if self.rank == 0:
            logger.debug(message)

    def _log_info(self, message: str):
        if self.rank == 0:
            logger.info(message)

    def _print(self, message: str):
        if self.rank == 0:
            print(message)


class Stepper(LoggingMixin):
    def __init__(self, rank: int = 0):
        self.rank: int = rank

    @property
    def _state(self):
        return DerivedFV3State(self._fv3gfs)

    def _compute_python_tendency(self) -> Diagnostics:
        return {}

    def _apply_python_to_dycore_state(self) -> Diagnostics:
        return {}

    def _apply_python_to_physics_state(self) -> Diagnostics:
        return {}


class TimeLoop(Iterable[Tuple[cftime.DatetimeJulian, Diagnostics]], LoggingMixin):
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

    def __init__(
        self,
        config: Optional[Mapping],
        comm: Any = None,
        fv3gfs: Any = wrapper,
        util=util,
    ) -> None:

        config = config or {}

        if comm is None:
            comm = MPI.COMM_WORLD

        self._fv3gfs = fv3gfs
        self._state: DerivedFV3State = DerivedFV3State(self._fv3gfs)
        self._comm = comm
        self._timer = util.Timer()
        self.rank: int = comm.rank

        namelist = runtime.get_namelist()

        # get timestep
        timestep = namelist["coupler_nml"]["dt_atmos"]
        self._timestep = timestep
        self._log_info(f"Timestep: {timestep}")

        self._tendencies_to_apply_to_dycore_state: State = {}
        self._tendencies_to_apply_to_physics_state: State = {}

        self._states_to_output: Sequence[str] = self._get_states_to_output(config)
        self._log_debug(f"States to output: {self._states_to_output}")
        self.stepper: Stepper = self._get_stepper(config)
        self._log_info(self._fv3gfs.get_tracer_metadata())
        MPI.COMM_WORLD.barrier()  # wait for initialization to finish

    def _get_states_to_output(self, config: Any) -> Sequence[str]:
        states_to_output = []
        for diagnostic in config.get("diagnostics", []):
            if diagnostic["name"] == "state_after_timestep.zarr":
                states_to_output = diagnostic["variables"]
        return states_to_output

    def _get_stepper(self, config: Mapping) -> Stepper:

        if "scikit_learn" in config:
            # download the model
            self._log_info("Downloading ML Model")
            model = open_model(config)
            self._log_info("Model Downloaded")
            do_only_diagnostic_ml: bool = config["scikit_learn"].get(
                "diagnostic_ml", False
            )
            return MLStepper(
                self._fv3gfs,
                self.rank,
                self._timestep,
                states_to_output=self._states_to_output,
                model=model,
                diagnostic_only=do_only_diagnostic_ml,
            )
        elif "nudging" in config:
            return NudgingStepper(
                self._fv3gfs,
                self._comm,
                config,
                timestep=self._timestep,
                states_to_output=self._states_to_output,
            )
        else:
            return BaselineStepper(self._fv3gfs, self.rank, self._states_to_output)

    @property
    def time(self) -> cftime.DatetimeJulian:
        return self._state.time

    def cleanup(self):
        self._print_global_timings()
        self._fv3gfs.cleanup()

    def _step_dynamics(self) -> Diagnostics:
        self._log_debug(f"Dynamics Step")
        self._fv3gfs.step_dynamics()
        # no diagnostics are computed by default
        return {}

    def _compute_physics(self) -> Diagnostics:
        self._log_debug(f"Physics Step (compute)")
        self._fv3gfs.compute_physics()
        # no diagnostics are computed by default
        return {}

    @property
    def _water_species(self) -> List[str]:
        a = self._fv3gfs.get_tracer_metadata()
        return [name for name in a if a[name]["is_water"]]

    def _apply_physics(self) -> Diagnostics:
        self._log_debug(f"Physics Step (apply)")
        self._fv3gfs.apply_physics()

        micro = self._fv3gfs.get_diagnostic_by_name(
            "tendency_of_specific_humidity_due_to_microphysics"
        ).data_array
        delp = self._state[DELP]
        return {
            "storage_of_specific_humidity_path_due_to_microphysics": (micro * delp).sum(
                "z"
            )
            / gravity,
            "evaporation": self._state["evaporation"],
            "cnvprcp_after_physics": self._fv3gfs.get_diagnostic_by_name(
                "cnvprcp"
            ).data_array,
            "total_precip_after_physics": self._state[TOTAL_PRECIP],
        }

    def _print_timing(self, name, min_val, max_val, mean_val):
        self._print(f"{name:<30}{min_val:15.4f}{max_val:15.4f}{mean_val:15.4f}")

    def _print_global_timings(self, root=0):
        is_root = self._comm.Get_rank() == root
        recvbuf = np.array(0.0)
        reduced = {}
        self._print("-----------------------------------------------------------------")
        self._print("         Reporting clock statistics from python runfile          ")
        self._print("-----------------------------------------------------------------")
        self._print(f"{' ':<30}{'min (s)':>15}{'max (s)':>15}{'mean (s)':>15}")
        for name, value in self._timer.times.items():
            reduced[name] = {}
            for label, op in [("min", MPI.MIN), ("max", MPI.MAX), ("mean", MPI.SUM)]:
                comm.Reduce(np.array(value), recvbuf, op=op)
                if is_root and label == "mean":
                    recvbuf /= comm.Get_size()
                reduced[name][label] = recvbuf.copy().item()
            self._print_timing(
                name, reduced[name]["min"], reduced[name]["max"], reduced[name]["mean"]
            )
        self._log_info(f"python_timing:{json.dumps(reduced)}")

    @property
    def _substeps(self) -> Sequence[Callable[..., Diagnostics]]:
        return [
            self._step_dynamics,
            self._compute_physics,
            self._apply_python_to_physics_state,
            self._apply_physics,
            self._compute_python_tendency,
            self._apply_python_to_dycore_state,
        ]

    def _apply_python_to_physics_state(self) -> Diagnostics:
        return self.stepper._apply_python_to_physics_state()

    def _compute_python_tendency(self) -> Diagnostics:
        return self.stepper._compute_python_tendency()

    def _apply_python_to_dycore_state(self) -> Diagnostics:
        return self.stepper._apply_python_to_dycore_state()

    def __iter__(self):
        for i in range(self._fv3gfs.get_step_count()):
            diagnostics = {}
            for substep in self._substeps:
                with self._timer.clock(substep.__name__):
                    diagnostics.update(substep())
            yield self._state.time, diagnostics


class BaselineStepper(Stepper):
    def __init__(self, fv3gfs, rank, states_to_output):
        self._fv3gfs = fv3gfs
        self._states_to_output = states_to_output

    def _compute_python_tendency(self) -> Diagnostics:
        return {}

    def _apply_python_to_dycore_state(self) -> Diagnostics:

        state: State = {name: self._state[name] for name in [PRECIP_RATE, SPHUM, DELP]}
        diagnostics: Diagnostics = runtime.compute_baseline_diagnostics(state)
        diagnostics.update({name: self._state[name] for name in self._states_to_output})

        return {
            "area": self._state[AREA],
            "cnvprcp_after_python": self._fv3gfs.get_diagnostic_by_name(
                "cnvprcp"
            ).data_array,
            **diagnostics,
        }


class MLStepper(Stepper):
    def __init__(
        self,
        fv3gfs: Any,
        rank: int,
        timestep: float,
        states_to_output: Any,
        model: fv3fit.Predictor,
        diagnostic_only: bool = False,
    ):
        self.rank: int = rank
        self._fv3gfs: Any = fv3gfs
        self._do_only_diagnostic_ml: bool = diagnostic_only
        self._timestep: float = timestep
        self._model: fv3fit.Predictor = model
        self._states_to_output = states_to_output

        self._tendencies_to_apply_to_dycore_state: State = {}
        self._tendencies_to_apply_to_physics_state: State = {}

    def _apply_python_to_dycore_state(self) -> Diagnostics:

        updated_state: State = {}

        variables: List[Hashable] = [
            TENDENCY_TO_STATE_NAME["dQ1"],
            TENDENCY_TO_STATE_NAME["dQ2"],
            DELP,
            PRECIP_RATE,
            TOTAL_PRECIP,
        ]
        self._log_debug(f"Getting state variables: {variables}")
        state = {name: self._state[name] for name in variables}
        tendency = self._tendencies_to_apply_to_dycore_state
        diagnostics = runtime.compute_ml_diagnostics(state, tendency)

        if self._do_only_diagnostic_ml:
            runtime.rename_diagnostics(diagnostics)
        else:
            updated_state.update(apply(state, tendency, dt=self._timestep))

        updated_state[TOTAL_PRECIP] = precipitation_sum(
            state[TOTAL_PRECIP], diagnostics["net_moistening"], self._timestep
        )

        self._log_debug("Setting Fortran State")
        self._state.update(updated_state)

        diagnostics.update({name: self._state[name] for name in self._states_to_output})

        return {
            "area": self._state[AREA],
            "cnvprcp_after_python": self._fv3gfs.get_diagnostic_by_name(
                "cnvprcp"
            ).data_array,
            "total_precip": updated_state[TOTAL_PRECIP],
            **diagnostics,
        }

    def _apply_python_to_physics_state(self) -> Diagnostics:
        self._log_debug(f"Apply python tendencies to physics state")
        variables: List[Hashable] = [
            TENDENCY_TO_STATE_NAME["dQu"],
            TENDENCY_TO_STATE_NAME["dQv"],
            DELP,
        ]
        state = {name: self._state[name] for name in variables}
        tendency = self._tendencies_to_apply_to_physics_state
        updated_state: State = apply(state, tendency, dt=self._timestep)
        diagnostics: Diagnostics = runtime.compute_ml_momentum_diagnostics(
            state, tendency
        )
        if self._do_only_diagnostic_ml:
            runtime.rename_diagnostics(diagnostics)
        else:
            self._state.update(updated_state)

        return diagnostics

    def _compute_python_tendency(self) -> Diagnostics:
        variables: List[Hashable] = list(set(self._model.input_variables) | {SPHUM})
        self._log_debug(f"Getting state variables: {variables}")
        state = {name: self._state[name] for name in variables}

        self._log_debug("Computing ML-predicted tendencies")
        tendency = predict(self._model, state)

        self._log_debug(
            "Correcting ML tendencies that would predict negative specific humidity"
        )
        tendency = limit_sphum_tendency(state, tendency, dt=self._timestep)

        self._tendencies_to_apply_to_dycore_state = {
            k: v for k, v in tendency.items() if k in ["dQ1", "dQ2"]
        }
        self._tendencies_to_apply_to_physics_state = {
            k: v for k, v in tendency.items() if k in ["dQu", "dQv"]
        }
        return {}


class NudgingStepper(Stepper):
    """Stepper for nudging
    """

    def __init__(
        self,
        fv3gfs: Any,
        comm: Any,
        config: Mapping,
        timestep: float,
        states_to_output: Any,
    ):

        self._states_to_output = states_to_output

        self._fv3gfs = fv3gfs
        self._comm = comm
        self.rank: int = comm.rank
        self._timestep: float = timestep

        self._nudging_timescales = runtime.nudging_timescales_from_dict(
            config["nudging"]["timescale_hours"]
        )
        self._get_reference_state = runtime.setup_get_reference_state(
            config,
            self.nudging_variables + [SST_NAME, TSFC_NAME],
            self._comm,
            self._fv3gfs.get_tracer_metadata(),
        )
        self._get_nudging_tendency = functools.partial(
            runtime.get_nudging_tendency, nudging_timescales=self._nudging_timescales,
        )
        self._tendencies_to_apply_to_dycore_state: State = {}

    @property
    def nudging_variables(self) -> List[str]:
        return list(self._nudging_timescales)

    def _compute_python_tendency(self) -> Diagnostics:

        self._log_debug("Computing nudging tendencies")
        variables: List[str] = self.nudging_variables + [
            SST_NAME,
            TSFC_NAME,
            MASK_NAME,
        ]
        state: State = {name: self._state[name] for name in variables}
        reference = self._get_reference_state(self._state.time)
        runtime.set_state_sst_to_reference(state, reference)
        self._tendencies_to_apply_to_dycore_state = self._get_nudging_tendency(
            state, reference
        )

        return {
            f"{key}_reference": reference_state
            for key, reference_state in reference.items()
        }

    def _apply_python_to_dycore_state(self) -> Diagnostics:

        diagnostics: Diagnostics = {}

        variables: List[str] = self.nudging_variables + [
            TOTAL_PRECIP,
            PRECIP_RATE,
            DELP,
        ]
        self._log_debug(f"Getting state variables: {variables}")
        state: State = {name: self._state[name] for name in variables}
        tendency: State = self._tendencies_to_apply_to_dycore_state

        diagnostics.update(runtime.compute_nudging_diagnostics(state, tendency))
        updated_state: State = apply(state, tendency, dt=self._timestep)
        updated_state[TOTAL_PRECIP] = precipitation_sum(
            state[TOTAL_PRECIP],
            diagnostics["net_moistening_due_to_nudging"],
            self._timestep,
        )

        self._log_debug("Setting Fortran State")
        self._state.update(updated_state)

        diagnostics.update({name: self._state[name] for name in self._states_to_output})

        return {
            "area": self._state[AREA],
            "cnvprcp_after_python": self._fv3gfs.get_diagnostic_by_name(
                "cnvprcp"
            ).data_array,
            "total_precip": updated_state[TOTAL_PRECIP],
            **diagnostics,
        }


def monitor(name: str, func):
    """Decorator to add tendency monitoring to an update function

    This will add the following diagnostics:
    - `tendency_of_{variable}_due_to_{name}`
    - `storage_of_{variable}_path_due_to_{name}`. A pressure-integrated version of the
       above
    - `storage_of_mass_due_to_{name}`, the total mass tendency in Pa/s.

    Args:
        name: the name to tag the tendency diagnostics with
        func: a stepping function, usually a bound method of TimeLoop

    Returns:
        monitored function. Same as func, but with tendency and mass change
        diagnostics.
    """

    def step(self: TimeLoop) -> Mapping[str, xr.DataArray]:

        vars_ = list(set(self._tendency_variables) | set(self._storage_variables))
        delp_before = self._state[DELP]
        before = {key: self._state[key] for key in vars_}

        diags = func(self)

        delp_after = self._state[DELP]
        after = {key: self._state[key] for key in vars_}

        # Compute statistics
        for variable in self._tendency_variables:
            diags[f"tendency_of_{variable}_due_to_{name}"] = (
                after[variable] - before[variable]
            ) / self._timestep

        for variable in self._storage_variables:
            path_before = (before[variable] * delp_before).sum("z") / gravity
            path_after = (after[variable] * delp_after).sum("z") / gravity

            diags[f"storage_of_{variable}_path_due_to_{name}"] = (
                path_after - path_before
            ) / self._timestep

        mass_change = (delp_after - delp_before).sum("z") / self._timestep
        mass_change.attrs["units"] = "Pa/s"
        diags[f"storage_of_mass_due_to_{name}"] = mass_change

        return diags

    # ensure monitored function has same name as original
    step.__name__ = func.__name__
    return step


class MonitoredPhysicsTimeLoop(TimeLoop):
    def __init__(
        self,
        tendency_variables: Sequence[str],
        storage_variables: Sequence[str],
        *args,
        **kwargs,
    ):
        """

        Args:
            tendency_variables: a list of variables to compute the physics
                tendencies of.

        """
        super().__init__(*args, **kwargs)
        self._tendency_variables = list(tendency_variables)
        self._storage_variables = list(storage_variables)

    _apply_physics = monitor("fv3_physics", TimeLoop._apply_physics)
    _apply_python_to_dycore_state = monitor(
        "python", TimeLoop._apply_python_to_dycore_state
    )


def globally_average_2d_diagnostics(
    comm,
    diagnostics: Mapping[str, xr.DataArray],
    exclude: Optional[Sequence[str]] = None,
) -> Mapping[str, float]:
    averages = {}
    exclude = exclude or []
    for v in diagnostics:
        if (set(diagnostics[v].dims) == {"x", "y"}) and (v not in exclude):
            averages[v] = global_average(comm, diagnostics[v], diagnostics["area"])
    return averages


if __name__ == "__main__":
    comm = MPI.COMM_WORLD

    config = runtime.get_config()
    partitioner = util.CubedSpherePartitioner.from_namelist(config["namelist"])
    setup_metrics_logger()

    loop = MonitoredPhysicsTimeLoop(
        config=config,
        comm=comm,
        tendency_variables=config.get("step_tendency_variables", []),
        storage_variables=config.get("step_storage_variables", []),
    )

    diag_files = runtime.get_diagnostic_files(
        config, partitioner, comm, initial_time=loop.time
    )

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
