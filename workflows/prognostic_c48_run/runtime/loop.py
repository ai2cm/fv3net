import datetime
import json
import logging
from typing import Any, Callable, Iterable, List, Mapping, Optional, Sequence, Tuple

import cftime
import fv3gfs.util
import fv3gfs.wrapper
import numpy as np
import xarray as xr
from mpi4py import MPI
from runtime import DerivedFV3State
from runtime.config import UserConfig, get_namelist
from runtime.diagnostics.machine_learning import (
    compute_baseline_diagnostics,
    rename_diagnostics,
)
from runtime.diagnostics.machine_learning import (
    precipitation_rate,
    precipitation_sum,
)
from runtime.steppers.machine_learning import PureMLStepper, open_model, download_model
from runtime.steppers.nudging import PureNudger
from runtime.types import Diagnostics, State, Tendencies
from runtime.names import TENDENCY_TO_STATE_NAME
from toolz import dissoc
from typing_extensions import Protocol

from .names import AREA, DELP, TOTAL_PRECIP

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


class Stepper(Protocol):
    """Stepper interface

    Steppers know the difference between tendencies, diagnostics, and
    in-place state updates, but they do not know how and when these updates
    will be applied.

    Note:
        Uses typing_extensions.Protocol to avoid the need for explicit sub-typing

    """

    @property
    def net_moistening(self) -> str:
        """Legacy variable needed to renaming the "net_moistening" variable

        This should be refactored away
        """
        pass

    def __call__(self, time, state) -> Tuple[Tendencies, Diagnostics, State]:
        return {}, {}, {}

    def get_diagnostics(self, state, tendency) -> Diagnostics:
        """Legacy method to provide polymorphic interface for diagnostics

        This should be refactored away by merging the diagnostics code, which
        is mostly very similar
        """
        return {}

    def get_momentum_diagnostics(self, state, tendency) -> Diagnostics:
        """Legacy method to provide polymorphic interface for diagnostics

        This should be refactored away by merging the diagnostics code, which
        is mostly very similar
        """
        return {}


def add_tendency(state: Any, tendency: State, dt: float) -> State:
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


class TimeLoop(Iterable[Tuple[cftime.DatetimeJulian, Diagnostics]], LoggingMixin):
    """An iterable defining the master time loop of a prognostic simulation

    Yields (time, diagnostics) tuples, which can be saved using diagnostic routines.

    Each time step of the model evolutions proceeds like this::

        step_dynamics,
        compute_physics,
        apply_python_to_physics_state
        apply_physics
        compute_python_updates
        apply_python_to_dycore_state

    The time loop relies on objects implementing the :py:class:`Stepper`
    interface to enable ML and other updates. The steppers compute their
    updates in ``_compute_python_updates``. The ``TimeLoop`` controls when
    and how to apply these updates to the FV3 state.
    """

    def __init__(
        self, config: UserConfig, comm: Any = None, wrapper: Any = fv3gfs.wrapper,
    ) -> None:

        if comm is None:
            comm = MPI.COMM_WORLD

        self._fv3gfs = wrapper
        self._state: DerivedFV3State = DerivedFV3State(self._fv3gfs)
        self.comm = comm
        self._timer = fv3gfs.util.Timer()
        self.rank: int = comm.rank

        namelist = get_namelist()

        # get timestep
        timestep = namelist["coupler_nml"]["dt_atmos"]
        self._timestep = timestep
        self._log_info(f"Timestep: {timestep}")

        self._do_only_diagnostic_ml = config.scikit_learn.diagnostic_ml
        self._tendencies: Tendencies = {}
        self._state_updates: State = {}

        self._states_to_output: Sequence[str] = self._get_states_to_output(config)
        self._log_debug(f"States to output: {self._states_to_output}")
        self.stepper = self._get_stepper(config)
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

    def _get_stepper(self, config: UserConfig) -> Optional[Stepper]:
        if config.scikit_learn.model:
            self._log_info("Using MLStepper")
            self._log_info("Downloading ML Model")
            if self.rank == 0:
                local_model_paths = download_model(config.scikit_learn, "ml_model")
            else:
                local_model_paths = None
            local_model_paths = self.comm.bcast(local_model_paths, root=0)
            setattr(config.scikit_learn, "model", local_model_paths)
            self._log_info("Model Downloaded From Remote")
            model = open_model(config.scikit_learn)
            self._log_info("Model Loaded")
            return PureMLStepper(model, self._timestep)
        elif config.nudging:
            self._log_info("Using NudgingStepper")
            partitioner = fv3gfs.util.CubedSpherePartitioner.from_namelist(
                get_namelist()
            )
            communicator = fv3gfs.util.CubedSphereCommunicator(self.comm, partitioner)
            return PureNudger(config.nudging, communicator)
        else:
            self._log_info("Performing baseline simulation")
            return None

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
                self.comm.Reduce(np.array(value), recvbuf, op=op)
                if is_root and label == "mean":
                    recvbuf /= self.comm.Get_size()
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
            self._compute_python_updates,
            self._apply_python_to_dycore_state,
        ]

    def _apply_python_to_physics_state(self) -> Diagnostics:
        """Apply computed tendencies and state updates to the physics state

        Mostly used for updating the eastward and northward winds.
        """
        self._log_debug(f"Apply python tendencies to physics state")
        tendency = {k: v for k, v in self._tendencies.items() if k in ["dQu", "dQv"]}

        diagnostics: Diagnostics = {}

        if self.stepper is not None:
            diagnostics = self.stepper.get_momentum_diagnostics(self._state, tendency)
            if self._do_only_diagnostic_ml:
                rename_diagnostics(diagnostics)
            else:
                updated_state = add_tendency(self._state, tendency, dt=self._timestep)
                self._state.update_mass_conserving(updated_state)

        return diagnostics

    def _compute_python_updates(self) -> Diagnostics:
        self._log_info("Computing Python Updates")

        if self.stepper is None:
            return {}
        else:
            (self._tendencies, diagnostics, self._state_updates,) = self.stepper(
                self._state.time, self._state
            )
            try:
                rank_updated_points = diagnostics["rank_updated_points"]
            except KeyError:
                pass
            else:
                updated_points = self.comm.reduce(rank_updated_points, root=0)
                if self.comm.rank == 0:
                    level_updates = {
                        i: int(value)
                        for i, value in enumerate(updated_points.sum(["x", "y"]).values)
                    }
                    logger.info(
                        f"specific_humidity_limiter_updates_per_level: {level_updates}"
                    )
            return diagnostics

    def _apply_python_to_dycore_state(self) -> Diagnostics:

        tendency = dissoc(self._tendencies, "dQu", "dQv")

        if self.stepper is None:
            diagnostics = compute_baseline_diagnostics(self._state)
        else:
            diagnostics = self.stepper.get_diagnostics(self._state, tendency)
            if self._do_only_diagnostic_ml:
                rename_diagnostics(diagnostics)
            else:
                updated_state = add_tendency(self._state, tendency, dt=self._timestep)
                updated_state[TOTAL_PRECIP] = precipitation_sum(
                    self._state[TOTAL_PRECIP],
                    diagnostics[self.stepper.net_moistening],
                    self._timestep,
                )
                diagnostics[TOTAL_PRECIP] = updated_state[TOTAL_PRECIP]
                self._state.update_mass_conserving(updated_state)
                self._state.update_mass_conserving(self._state_updates)

        diagnostics.update({name: self._state[name] for name in self._states_to_output})
        diagnostics.update(
            {
                "area": self._state[AREA],
                "cnvprcp_after_python": self._fv3gfs.get_diagnostic_by_name(
                    "cnvprcp"
                ).data_array,
                "total_precipitation_rate": precipitation_rate(
                    self._state[TOTAL_PRECIP], self._timestep
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
