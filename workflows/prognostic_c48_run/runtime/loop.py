import cftime
import datetime
import json
import logging
from typing import (
    Any,
    Callable,
    Iterable,
    Mapping,
    Tuple,
    List,
    Sequence,
    Optional,
)

import numpy as np
import xarray as xr
from mpi4py import MPI

from runtime import DerivedFV3State
import fv3gfs.wrapper
import fv3gfs.util

from runtime.steppers.machine_learning import open_model, MLStepper
from runtime.config import UserConfig, get_namelist
from runtime.steppers.base import (
    LoggingMixin,
    Stepper,
    BaselineStepper,
    State,
    Diagnostics,
)

from runtime.steppers.nudging import NudgingStepper
from runtime.steppers.base import precipitation_rate

from .names import (
    TOTAL_PRECIP,
    AREA,
    DELP,
)


logger = logging.getLogger(__name__)

gravity = 9.81


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
        self, config: UserConfig, comm: Any = None, wrapper: Any = fv3gfs.wrapper,
    ) -> None:

        if comm is None:
            comm = MPI.COMM_WORLD

        self._fv3gfs = wrapper
        self._state: DerivedFV3State = DerivedFV3State(self._fv3gfs)
        self._comm = comm
        self._timer = fv3gfs.util.Timer()
        self.rank: int = comm.rank

        namelist = get_namelist()

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

    def _get_states_to_output(self, config: UserConfig) -> Sequence[str]:
        states_to_output: List[str] = []
        for diagnostic in config.diagnostics:
            if diagnostic.name == "state_after_timestep.zarr":
                if diagnostic.variables:
                    # states_to_output should be a Container but fixing this
                    # type error requires changing its usage by the steppers
                    states_to_output = diagnostic.variables  # type: ignore
        return states_to_output

    def _get_stepper(self, config: UserConfig) -> Stepper:

        if config.scikit_learn.model:
            self._log_info("Using MLStepper")
            # download the model
            self._log_info("Downloading ML Model")
            model = open_model(config.scikit_learn)
            self._log_info("Model Downloaded")
            return MLStepper(
                self._state,
                self._comm,
                self._timestep,
                states_to_output=self._states_to_output,
                model=model,
                diagnostic_only=config.scikit_learn.diagnostic_ml,
            )
        elif config.nudging:
            self._log_info("Using NudgingStepper")
            partitioner = fv3gfs.util.CubedSpherePartitioner.from_namelist(
                get_namelist()
            )
            communicator = fv3gfs.util.CubedSphereCommunicator(self._comm, partitioner)
            return NudgingStepper(
                self._state,
                self._fv3gfs,
                self._comm.rank,
                config.nudging,
                timestep=self._timestep,
                communicator=communicator,
            )
        else:
            self._log_info("Using BaselineStepper")
            return BaselineStepper(self._state)

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
        is_root = self.rank == root
        recvbuf = np.array(0.0)
        reduced = {}
        self._print("-----------------------------------------------------------------")
        self._print("         Reporting clock statistics from python runfile          ")
        self._print("-----------------------------------------------------------------")
        self._print(f"{' ':<30}{'min (s)':>15}{'max (s)':>15}{'mean (s)':>15}")
        for name, value in self._timer.times.items():
            reduced[name] = {}
            for label, op in [("min", MPI.MIN), ("max", MPI.MAX), ("mean", MPI.SUM)]:
                self._comm.Reduce(np.array(value), recvbuf, op=op)
                if is_root and label == "mean":
                    recvbuf /= self._comm.Get_size()
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

        diagnostics = self.stepper._apply_python_to_dycore_state()
        diagnostics.update({name: self._state[name] for name in self._states_to_output})
        diagnostics.update(
            {
                "area": self._state[AREA],
                "cnvprcp_after_python": self._fv3gfs.get_diagnostic_by_name(
                    "cnvprcp"
                ).data_array,
                "total_precipitation_rate": precipitation_rate(
                    diagnostics[TOTAL_PRECIP], self._timestep
                ),
            }
        )

        try:
            del diagnostics[TOTAL_PRECIP]
        except KeyError:
            pass

        return diagnostics

    def __iter__(self):
        for i in range(self._fv3gfs.get_step_count()):
            diagnostics = {}
            for substep in self._substeps:
                with self._timer.clock(substep.__name__):
                    diagnostics.update(substep())
            yield self._state.time, diagnostics


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

    def step(self) -> Mapping[str, xr.DataArray]:

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
