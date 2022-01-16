import json
import logging
import os
import tempfile
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)
import cftime
import fv3gfs.util
import fv3gfs.wrapper
import numpy as np
import vcm
import xarray as xr
from mpi4py import MPI
import runtime.factories
from runtime.derived_state import DerivedFV3State, MergedState
from runtime.config import UserConfig, get_namelist
from runtime.diagnostics.compute import (
    compute_baseline_diagnostics,
    precipitation_rate,
    precipitation_sum,
    precipitation_accumulation,
    rename_diagnostics,
)
from runtime.monitor import Monitor
from runtime.names import (
    TENDENCY_TO_STATE_NAME,
    TOTAL_PRECIP_RATE,
    PREPHYSICS_OVERRIDES,
)
from runtime.steppers.machine_learning import (
    MachineLearningConfig,
    PureMLStepper,
    download_model,
    open_model,
)
from runtime.steppers.nudging import PureNudger
from runtime.steppers.prescriber import Prescriber, PrescriberConfig, get_timesteps
from runtime.steppers.combine import CombinedStepper
from runtime.types import Diagnostics, State, Tendencies, Step
from toolz import dissoc
from typing_extensions import Protocol

from .names import AREA, DELP, TOTAL_PRECIP

logger = logging.getLogger(__name__)


class Stepper(Protocol):
    """Stepper interface

    Steppers know the difference between tendencies, diagnostics, and
    in-place state updates, but they do not know how and when these updates
    will be applied.

    Note:
        Uses typing_extensions.Protocol to avoid the need for explicit sub-typing

    """

    @property
    def label(self) -> str:
        """Label used for naming diagnostics.
        """
        pass

    def __call__(self, time, state) -> Tuple[Tendencies, Diagnostics, State]:
        return {}, {}, {}

    def get_diagnostics(self, state, tendency) -> Tuple[Diagnostics, xr.DataArray]:
        """Return diagnostics mapping and net moistening array."""
        return {}, xr.DataArray()

    def get_momentum_diagnostics(self, state, tendency) -> Diagnostics:
        """Return diagnostics of momentum tendencies."""
        return {}


def _replace_precip_rate_with_accumulation(  # type: ignore
    state_updates: State, dt: float
) -> State:
    # Precipitative ML models predict a rate, but the precipitation to update
    # in the state is an accumulated total over the timestep
    if TOTAL_PRECIP_RATE in state_updates:
        state_updates[TOTAL_PRECIP] = precipitation_accumulation(
            state_updates[TOTAL_PRECIP_RATE], dt
        )
        state_updates.pop(TOTAL_PRECIP_RATE)


def add_tendency(state: Any, tendency: State, dt: float) -> State:
    """Given state and tendency prediction, return updated state.
    Returned state only includes variables updated by ML model."""

    with xr.set_options(keep_attrs=True):
        updated = {}
        for name_ in tendency:
            name = str(name_)
            try:
                state_name = str(TENDENCY_TO_STATE_NAME[name])
            except KeyError:
                raise KeyError(
                    f"Tendency variable '{name}' does not have an entry mapping it "
                    "to a corresponding state variable to add to. "
                    "Existing tendencies with mappings to state are "
                    f"{list(TENDENCY_TO_STATE_NAME.keys())}"
                )
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


class TimeLoop(
    Iterable[Tuple[cftime.DatetimeJulian, Dict[str, xr.DataArray]]], LoggingMixin
):
    """An iterable defining the master time loop of a prognostic simulation

    Yields (time, diagnostics) tuples, which can be saved using diagnostic routines.

    Each time step of the model evolutions proceeds like this::

        step_dynamics,
        step_prephysics,
        compute_physics,
        apply_postphysics_to_physics_state,
        apply_physics,
        compute_postphysics,
        apply_postphysics_to_dycore_state,

    The time loop relies on objects implementing the :py:class:`Stepper`
    interface to enable ML and other updates. The steppers compute their
    updates in ``_step_prephysics`` and ``_compute_postphysics``. The
    ``TimeLoop`` controls when and how to apply these updates to the FV3 state.
    """

    def __init__(
        self, config: UserConfig, comm: Any = None, wrapper: Any = fv3gfs.wrapper,
    ) -> None:

        if comm is None:
            comm = MPI.COMM_WORLD

        self._fv3gfs = wrapper
        self._state: DerivedFV3State = MergedState(DerivedFV3State(self._fv3gfs), {})
        self.comm = comm
        self._timer = fv3gfs.util.Timer()
        self.rank: int = comm.rank

        namelist = get_namelist()

        # get timestep
        timestep = namelist["coupler_nml"]["dt_atmos"]
        self._timestep = timestep
        self._log_info(f"Timestep: {timestep}")
        hydrostatic = namelist["fv_core_nml"]["hydrostatic"]

        self._prephysics_only_diagnostic_ml: bool = self._use_diagnostic_ml_prephysics(
            getattr(config, "prephysics")
        )
        self._postphysics_only_diagnostic_ml: bool = getattr(
            getattr(config, "scikit_learn"), "diagnostic_ml", False
        )
        self._tendencies: Tendencies = {}
        self._state_updates: State = {}

        self.monitor = Monitor.from_variables(
            config.diagnostic_variables, state=self._state, timestep=self._timestep,
        )
        self._transform_physics = runtime.factories.get_fv3_physics_transformer(
            config, self._state, self._timestep,
        )
        self._prescribe_tendency = runtime.factories.get_tendency_prescriber(
            config, self._state, self._timestep, self._get_communicator(),
        )

        self._states_to_output: Sequence[str] = self._get_states_to_output(config)
        self._log_debug(f"States to output: {self._states_to_output}")
        self._prephysics_stepper = self._get_prephysics_stepper(config, hydrostatic)
        self._postphysics_stepper = self._get_postphysics_stepper(config, hydrostatic)
        self._log_info(self._fv3gfs.get_tracer_metadata())
        MPI.COMM_WORLD.barrier()  # wait for initialization to finish

    def _use_diagnostic_ml_prephysics(self, prephysics_config):
        if prephysics_config is None:
            return False
        diag_ml_usages = sum(
            [getattr(c, "diagnostic_ml", False) for c in prephysics_config]
        )
        if diag_ml_usages == 0:
            return False
        elif diag_ml_usages == 1:
            return True
        else:
            raise ValueError(
                "If multiple ML models are provided in config.prephysics, "
                "all must have same values for diagnostic_ml."
            )

    @staticmethod
    def _get_states_to_output(config: UserConfig) -> Sequence[str]:
        states_to_output: List[str] = []
        for diagnostic in config.diagnostics:
            if diagnostic.name == "state_after_timestep.zarr":
                if diagnostic.variables:
                    # states_to_output should be a Container but fixing this
                    # type error requires changing its usage by the steppers
                    states_to_output = diagnostic.variables  # type: ignore
        return states_to_output

    def _get_communicator(self):
        partitioner = fv3gfs.util.CubedSpherePartitioner.from_namelist(get_namelist())
        return fv3gfs.util.CubedSphereCommunicator(self.comm, partitioner)

    def emulate_or_prescribe_tendency(self, func: Step) -> Step:
        if self._transform_physics is not None and self._prescribe_tendency is not None:
            return self._prescribe_tendency(self._transform_physics(func))
        elif self._transform_physics is None and self._prescribe_tendency is not None:
            return self._prescribe_tendency(func)
        elif self._transform_physics is not None and self._prescribe_tendency is None:
            return self._transform_physics(func)
        else:
            return func

    def _get_prephysics_stepper(
        self, config: UserConfig, hydrostatic: bool
    ) -> Optional[Stepper]:
        stepper: Optional[Stepper]
        if config.prephysics is None:
            self._log_info("No prephysics computations")
            stepper = None

        else:
            prephysics_steppers: List[Union[Prescriber, PureMLStepper]] = []
            for prephysics_config in config.prephysics:
                if isinstance(prephysics_config, MachineLearningConfig):
                    self._log_info("Using PureMLStepper for prephysics")
                    model = self._open_model(prephysics_config, "_prephysics")
                    prephysics_steppers.append(
                        PureMLStepper(model, self._timestep, hydrostatic)
                    )
                elif isinstance(prephysics_config, PrescriberConfig):
                    self._log_info(
                        "Using Prescriber for prephysics for variables "
                        f"{prephysics_config.variables}"
                    )
                    communicator = self._get_communicator()
                    timesteps = get_timesteps(
                        self.time, self._timestep, self._fv3gfs.get_step_count()
                    )
                    prephysics_steppers.append(
                        Prescriber(prephysics_config, communicator, timesteps=timesteps)
                    )
            stepper = CombinedStepper(prephysics_steppers)
        return stepper

    def _get_postphysics_stepper(
        self, config: UserConfig, hydrostatic: bool
    ) -> Optional[Stepper]:
        if config.scikit_learn:
            self._log_info("Using MLStepper for postphysics updates")
            model = self._open_model(config.scikit_learn, "_postphysics")
            stepper: Optional[Stepper] = PureMLStepper(
                model, self._timestep, hydrostatic
            )
        elif config.nudging:
            self._log_info("Using NudgingStepper for postphysics updates")
            stepper = PureNudger(config.nudging, self._get_communicator(), hydrostatic)
        else:
            self._log_info("Performing baseline simulation")
            stepper = None
        return stepper

    def _open_model(self, ml_config: MachineLearningConfig, step: str):
        self._log_info("Downloading ML Model")
        with tempfile.TemporaryDirectory() as tmpdir:
            if self.rank == 0:
                local_model_paths = download_model(
                    ml_config, os.path.join(tmpdir, step)
                )
            else:
                local_model_paths = None  # type: ignore
            local_model_paths = self.comm.bcast(local_model_paths, root=0)
            setattr(ml_config, "model", local_model_paths)
            self._log_info("Model Downloaded From Remote")
            model = open_model(ml_config)
            MPI.COMM_WORLD.barrier()
        self._log_info("Model Loaded")
        return model

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
            "storage_of_specific_humidity_path_due_to_microphysics": vcm.mass_integrate(
                micro, delp, "z"
            ),
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
        self._print("         Reporting clock statistics from python                  ")
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

    def _step_prephysics(self) -> Diagnostics:

        if self._prephysics_stepper is None:
            diagnostics: Diagnostics = {}
        else:
            self._log_debug("Computing prephysics updates")
            _, diagnostics, state_updates = self._prephysics_stepper(
                self._state.time, self._state
            )
            if self._prephysics_only_diagnostic_ml:
                rename_diagnostics(diagnostics)
            else:
                self._state_updates.update(state_updates)
        state_updates = {
            k: v for k, v in self._state_updates.items() if k in PREPHYSICS_OVERRIDES
        }
        self._state_updates = dissoc(self._state_updates, *PREPHYSICS_OVERRIDES)
        self._log_debug(
            f"Applying prephysics state updates for: {list(state_updates.keys())}"
        )
        self._state.update_mass_conserving(state_updates)

        return diagnostics

    def _apply_postphysics_to_physics_state(self) -> Diagnostics:
        """Apply computed tendencies and state updates to the physics state

        Mostly used for updating the eastward and northward winds.
        """
        self._log_debug(f"Apply postphysics tendencies to physics state")
        tendency = {k: v for k, v in self._tendencies.items() if k in ["dQu", "dQv"]}

        diagnostics: Diagnostics = {}

        if self._postphysics_stepper is not None:
            diagnostics = self._postphysics_stepper.get_momentum_diagnostics(
                self._state, tendency
            )
            if self._postphysics_only_diagnostic_ml:
                rename_diagnostics(diagnostics)
            else:
                updated_state = add_tendency(self._state, tendency, dt=self._timestep)
                self._state.update_mass_conserving(updated_state)

        return diagnostics

    def _compute_postphysics(self) -> Diagnostics:
        self._log_info("Computing Postphysics Updates")

        if self._postphysics_stepper is None:
            return {}
        else:
            (self._tendencies, diagnostics, state_updates,) = self._postphysics_stepper(
                self._state.time, self._state
            )
            _replace_precip_rate_with_accumulation(state_updates, self._timestep)

            self._log_debug(
                "Postphysics stepper adds tendency update to state for "
                f"{self._tendencies.keys()}"
            )
            self._log_debug(
                "Postphysics stepper updates state directly for "
                f"{state_updates.keys()}"
            )
            self._state_updates.update(state_updates)

            return diagnostics

    def _apply_postphysics_to_dycore_state(self) -> Diagnostics:

        tendency = dissoc(self._tendencies, "dQu", "dQv")
        diagnostics = compute_baseline_diagnostics(self._state)

        if self._postphysics_stepper is not None:
            stepper_diags, net_moistening = self._postphysics_stepper.get_diagnostics(
                self._state, tendency
            )
            diagnostics.update(stepper_diags)
            if self._postphysics_only_diagnostic_ml:
                rename_diagnostics(diagnostics)
            else:
                updated_state_from_tendency = add_tendency(
                    self._state, tendency, dt=self._timestep
                )

                # if total precip is updated directly by stepper,
                # it will overwrite this precipitation_sum
                updated_state_from_tendency[TOTAL_PRECIP] = precipitation_sum(
                    self._state[TOTAL_PRECIP], net_moistening, self._timestep,
                )
                self._state.update_mass_conserving(updated_state_from_tendency)
        self._log_info(
            "Applying state updates to postphysics dycore state: "
            f"{self._state_updates.keys()}"
        )
        self._state.update_mass_conserving(self._state_updates)

        diagnostics.update({name: self._state[name] for name in self._states_to_output})
        diagnostics.update(
            {
                "area": self._state[AREA],
                "cnvprcp_after_python": self._fv3gfs.get_diagnostic_by_name(
                    "cnvprcp"
                ).data_array,
                TOTAL_PRECIP_RATE: precipitation_rate(
                    self._state[TOTAL_PRECIP], self._timestep
                ),
            }
        )
        return diagnostics

    def __iter__(
        self,
    ) -> Iterator[Tuple[cftime.DatetimeJulian, Dict[str, xr.DataArray]]]:

        for i in range(self._fv3gfs.get_step_count()):
            diagnostics: Diagnostics = {}
            for substep in [
                self.monitor("dynamics", self._step_dynamics),
                self._step_prephysics,
                self._compute_physics,
                self._apply_postphysics_to_physics_state,
                self.monitor(
                    "applied_physics",
                    self.emulate_or_prescribe_tendency(
                        self.monitor("fv3_physics", self._apply_physics)
                    ),
                ),
                self._compute_postphysics,
                self.monitor("python", self._apply_postphysics_to_dycore_state),
            ]:
                with self._timer.clock(substep.__name__):
                    diagnostics.update(substep())
            yield self._state.time, {str(k): v for k, v in diagnostics.items()}
