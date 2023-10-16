import json
import logging
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
    Mapping,
    Hashable,
)
import cftime
import pace.util
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
import runtime.diagnostics.tracers
from runtime.monitor import Monitor
from runtime.names import (
    TOTAL_PRECIP_RATE,
    PREPHYSICS_OVERRIDES,
    SURFACE_FLUX_OVERRIDES,
)
from runtime.tendency import (
    add_tendency,
    prepare_tendencies_for_dynamical_core,
    state_updates_from_tendency,
)
from runtime.steppers.machine_learning import (
    MachineLearningConfig,
    PureMLStepper,
    download_model,
    open_model,
)
from runtime.steppers.stepper import Stepper
from runtime.steppers.nudging import PureNudger
from runtime.steppers.prescriber import PrescriberConfig
from runtime.steppers.interval import IntervalStepper, IntervalConfig
from runtime.steppers.combine import CombinedStepper
from runtime.steppers.reservoir import get_reservoir_steppers
from runtime.types import Diagnostics, State, Tendencies, Step
from toolz import dissoc

from runtime.nudging import NudgingConfig

from .names import AREA, DELP, TOTAL_PRECIP

logger = logging.getLogger(__name__)


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


def _check_surface_flux_overrides_exist(namelist_override_flag, state_update_keys):
    if namelist_override_flag is True:
        if not set(SURFACE_FLUX_OVERRIDES).issubset(state_update_keys):
            raise ValueError(
                "Namelist flag 'override_surface_radiative_fluxes' is set to True."
                f"Surface flux overrides {SURFACE_FLUX_OVERRIDES} must be in "
                "prephysics updates."
            )


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

    def __init__(self, config: UserConfig, wrapper: Any, comm: Any = None,) -> None:

        if comm is None:
            comm = MPI.COMM_WORLD

        self._fv3gfs = wrapper
        self._state: DerivedFV3State = MergedState(DerivedFV3State(self._fv3gfs), {})
        self.comm = comm
        self._timer = pace.util.Timer()
        self.rank: int = comm.rank

        namelist = get_namelist()

        # get timestep
        timestep = namelist["coupler_nml"]["dt_atmos"]
        self._timestep = timestep
        self._log_info(f"Timestep: {timestep}")
        hydrostatic = namelist["fv_core_nml"]["hydrostatic"]
        init_time = cftime.DatetimeJulian(*namelist["coupler_nml"]["current_date"])

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
        self._radiation_stepper = self._get_radiation_stepper(config, namelist)
        [
            self._reservior_increment_stepper,
            self._reservoir_predict_stepper,
        ] = self._get_reservoir_stepper(config, init_time)
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
        partitioner = pace.util.CubedSpherePartitioner.from_namelist(get_namelist())
        return pace.util.CubedSphereCommunicator(self.comm, partitioner)

    def emulate_or_prescribe_tendency(self, func: Step) -> Step:
        if self._transform_physics is not None and self._prescribe_tendency is not None:
            return self._prescribe_tendency(self._transform_physics(func))
        elif self._transform_physics is None and self._prescribe_tendency is not None:
            return self._prescribe_tendency(func)
        elif self._transform_physics is not None and self._prescribe_tendency is None:
            return self._transform_physics(func)
        else:
            return func

    def _get_stepper(
        self,
        stepper_config: Union[
            PrescriberConfig, MachineLearningConfig, NudgingConfig, IntervalConfig,
        ],
        step: str,
        hydrostatic: bool = False,
    ) -> Stepper:
        stepper: Stepper
        if isinstance(stepper_config, IntervalConfig):
            base_stepper_config = stepper_config.base_config
        else:
            base_stepper_config = stepper_config

        if isinstance(base_stepper_config, MachineLearningConfig):
            model = self._open_model(base_stepper_config)
            self._log_info(f"Using PureMLStepper at {step}.")
            if base_stepper_config.use_mse_conserving_humidity_limiter:
                self._log_info(f"Using MSE-conserving moisture limiter for step {step}")
            else:
                self._log_info(
                    f"Using old non-MSE-conserving moisture limiter for step {step}"
                )
            limit_mse = base_stepper_config.use_mse_conserving_humidity_limiter
            stepper = PureMLStepper(
                model=model,
                timestep=self._timestep,
                hydrostatic=hydrostatic,
                mse_conserving_limiter=limit_mse,
            )
        elif isinstance(base_stepper_config, NudgingConfig):
            self._log_info(f"Using NudgingStepper for step {step}")
            stepper = PureNudger(
                self._fv3gfs, base_stepper_config, self._get_communicator(), hydrostatic
            )
        else:
            self._log_info(
                "Using Prescriber for variables "
                f"{list(base_stepper_config.variables.values())} at {step}."
            )
            stepper = runtime.factories.get_prescriber(
                base_stepper_config, self._get_communicator()
            )

        if isinstance(stepper_config, IntervalConfig):
            return IntervalStepper(
                apply_interval_seconds=stepper_config.apply_interval_seconds,
                stepper=stepper,
                offset_seconds=stepper_config.offset_seconds,
                record_fields_before_update=stepper_config.record_fields_before_update,
            )
        else:
            return stepper

    def _get_prephysics_stepper(
        self, config: UserConfig, hydrostatic: bool
    ) -> Optional[Stepper]:
        if config.prephysics is None:
            self._log_info("No prephysics computations")
            return None
        else:
            prephysics_steppers: List[Stepper] = []
            for prephysics_config in config.prephysics:
                prephysics_steppers.append(
                    self._get_stepper(prephysics_config, "prephysics", hydrostatic)
                )
            return CombinedStepper(prephysics_steppers)

    def _get_postphysics_stepper(
        self, config: UserConfig, hydrostatic: bool
    ) -> Optional[Stepper]:
        postphysics_configs = filter(
            None, [config.scikit_learn, config.nudging, config.bias_correction]
        )
        postphysics_steppers: List[Stepper] = []
        for postphysics_config in postphysics_configs:
            postphysics_steppers.append(
                self._get_stepper(
                    postphysics_config, "postphysics", hydrostatic  # type: ignore
                )
            )
        if len(postphysics_steppers) > 0:
            stepper = CombinedStepper(postphysics_steppers)
        else:
            self._log_info("Performing baseline simulation")
            stepper = None  # type: ignore

        return stepper

    def _get_radiation_stepper(
        self, config: UserConfig, namelist: Mapping[Hashable, Any]
    ) -> Optional[Stepper]:
        if config.radiation_scheme is not None:
            radiation_input_generator_config = config.radiation_scheme.input_generator
            if radiation_input_generator_config is not None:
                radiation_input_generator: Optional[Stepper] = self._get_stepper(
                    radiation_input_generator_config, "radiation_inputs"
                )
            else:
                radiation_input_generator = None
            stepper: Optional[Stepper] = runtime.factories.get_radiation_stepper(
                self.comm,
                namelist,
                self._timestep,
                self._state["initialization_time"].item(),
                self._fv3gfs.get_tracer_metadata(),
                radiation_input_generator,
            )
        else:
            stepper = None
        return stepper

    def _get_reservoir_stepper(
        self, config: UserConfig, init_time: cftime.DatetimeJulian,
    ) -> Tuple[Optional[Stepper], Optional[Stepper]]:
        if config.reservoir_corrector is not None:
            res_config = config.reservoir_corrector
            self._log_info("Getting reservoir steppers")
            incrementer, predictor = get_reservoir_steppers(
                res_config,
                MPI.COMM_WORLD.Get_rank(),
                init_time=init_time,
                model_timestep=self._timestep,
            )
        else:
            incrementer, predictor = None, None

        return incrementer, predictor

    def _open_model(self, ml_config: MachineLearningConfig):
        self._log_info("Downloading ML Model")
        with tempfile.TemporaryDirectory() as tmpdir:
            if self.rank == 0:
                local_model_paths = download_model(ml_config, tmpdir)
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

    def _step_dynamics(self) -> Diagnostics:
        self._log_debug(f"Dynamics Step")
        self._fv3gfs.step_dynamics()
        # no diagnostics are computed by default
        return {}

    def _step_pre_radiation_physics(self) -> Diagnostics:
        self._log_debug(f"Pre-radiation Physics Step")
        self._fv3gfs.step_pre_radiation()
        return {
            f"{name}_pre_radiation": self._state[name]
            for name in self._states_to_output
        }

    def _step_radiation_physics(self) -> Diagnostics:
        self._log_debug(f"Radiation Physics Step")
        if self._radiation_stepper is not None:
            _, diagnostics, _ = self._radiation_stepper(self.time, self._state,)
        else:
            diagnostics = {}
        self._fv3gfs.step_radiation()
        return diagnostics

    def _step_post_radiation_physics(self) -> Diagnostics:
        self._log_debug(f"Post-radiation Physics Step")
        self._fv3gfs.step_post_radiation_physics()
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

    def _print_timings(self, reduced):
        self._print("-----------------------------------------------------------------")
        self._print("         Reporting clock statistics from python                  ")
        self._print("-----------------------------------------------------------------")
        self._print(f"{' ':<30}{'min (s)':>15}{'max (s)':>15}{'mean (s)':>15}")
        for name, timing in reduced.items():
            self._print(
                f"{name:<30}{timing['min']:15.4f}"
                f"{timing['max']:15.4f}{timing['mean']:15.4f}"
            )

    def log_global_timings(self, root=0):
        is_root = self.rank == root
        recvbuf = np.array(0.0)
        reduced = {}
        for name, value in self._timer.times.items():
            reduced[name] = {}
            for label, op in [("min", MPI.MIN), ("max", MPI.MAX), ("mean", MPI.SUM)]:
                self.comm.Reduce(np.array(value), recvbuf, op=op)
                if is_root and label == "mean":
                    recvbuf /= self.comm.Get_size()
                reduced[name][label] = recvbuf.copy().item()
        self._print_timings(reduced)
        log_out = {
            "steps": reduced,
            "units": "[s], cumulative and reduced across ranks",
        }
        self._log_info(json.dumps({"python_timing": log_out}))

    def _step_prephysics(self) -> Diagnostics:

        if self._prephysics_stepper is None:
            diagnostics: Diagnostics = {}
        else:
            self._log_debug("Computing prephysics updates")
            _, diagnostics, state_updates = self._prephysics_stepper(
                self._state.time, self._state
            )
            if isinstance(self._prephysics_stepper, PureMLStepper) and diagnostics:
                self._log_debug(
                    f"Exposing ML predictands {list(diagnostics.keys())} from "
                    f"prephysics stepper as diagnostics because they are not "
                    f"state updates or tendencies"
                )
            if self._prephysics_only_diagnostic_ml:
                rename_diagnostics(diagnostics)
            else:
                self._state_updates.update(state_updates)
        self._log_info(
            f"State updates from prescriber: {list(self._state_updates.keys())}"
        )

        state_updates = {
            k: v for k, v in self._state_updates.items() if k in PREPHYSICS_OVERRIDES
        }
        _check_surface_flux_overrides_exist(
            self._fv3gfs.flags.override_surface_radiative_fluxes,
            list(state_updates.keys()),
        )
        self._state_updates = dissoc(self._state_updates, *PREPHYSICS_OVERRIDES)
        self._log_debug(
            f"Applying prephysics state updates for: {list(state_updates.keys())}"
        )
        self._state.update_mass_conserving(state_updates)

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
            if isinstance(self._postphysics_stepper, PureMLStepper) and diagnostics:
                self._log_debug(
                    f"Exposing ML predictands {list(diagnostics.keys())} from "
                    f"postphysics stepper as diagnostics because they are not "
                    f"state updates or tendencies"
                )
            self._state_updates.update(state_updates)

            return diagnostics

    def _apply_postphysics_to_dycore_state(self) -> Diagnostics:

        diagnostics = compute_baseline_diagnostics(self._state)

        if self._postphysics_stepper is not None:
            stepper_diags, net_moistening = self._postphysics_stepper.get_diagnostics(
                self._state, self._tendencies
            )
            diagnostics.update(stepper_diags)
            if self._postphysics_only_diagnostic_ml:
                rename_diagnostics(diagnostics)
            else:
                (
                    filled_tendencies,
                    tendencies_filled_frac,
                ) = prepare_tendencies_for_dynamical_core(
                    self._fv3gfs, self._tendencies
                )
                updated_state_from_tendency = add_tendency(
                    self._state, filled_tendencies, dt=self._timestep
                )

                # if total precip is updated directly by stepper,
                # it will overwrite this precipitation_sum
                updated_state_from_tendency[TOTAL_PRECIP] = precipitation_sum(
                    self._state[TOTAL_PRECIP], net_moistening, self._timestep,
                )
                diagnostics.update(
                    state_updates_from_tendency(updated_state_from_tendency)
                )
                self._state.update_mass_conserving(updated_state_from_tendency)
                diagnostics.update(tendencies_filled_frac)

        self._log_info(
            f"{self._state.time} Applying state updates to postphysics dycore state: "
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

    def _increment_reservoir(self) -> Diagnostics:
        if self._reservior_increment_stepper is not None:
            [_, diags, _] = self._reservior_increment_stepper(
                self._state.time, self._state
            )
            return diags
        else:
            return {}

    def _apply_reservoir_update_to_state(self) -> Diagnostics:
        # TODO: handle tendencies. Currently the returned tendencies
        # are only used for diagnostics and are not used in updating state
        if self._reservoir_predict_stepper is not None:
            (
                tendencies_from_state_prediction,
                diags,
                state_updates,
            ) = self._reservoir_predict_stepper(self._state.time, self._state)
            (
                stepper_diags,
                net_moistening,
            ) = self._reservoir_predict_stepper.get_diagnostics(
                self._state, tendencies_from_state_prediction
            )
            diags.update(stepper_diags)
            if self._reservoir_predict_stepper.is_diagnostic:  # type: ignore
                rename_diagnostics(diags, label="reservoir_predictor")

            state_updates[TOTAL_PRECIP] = precipitation_sum(
                self._state[TOTAL_PRECIP], net_moistening, self._timestep,
            )

            self._state.update_mass_conserving(state_updates)

            diags.update({name: self._state[name] for name in self._states_to_output})
            diags.update(
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

            return diags
        else:
            return {}

    def _intermediate_restarts(self) -> Diagnostics:
        self._log_info("Saving intermediate restarts if enabled.")
        self._fv3gfs.save_intermediate_restart_if_enabled()
        return {}

    def __iter__(
        self,
    ) -> Iterator[Tuple[cftime.DatetimeJulian, Dict[str, xr.DataArray]]]:

        for i in range(self._fv3gfs.get_step_count()):
            diagnostics: Diagnostics = {}
            # clear the state updates in case some updates are on intervals
            self._state_updates = {}
            for substep in [
                lambda: runtime.diagnostics.tracers.compute_column_integrated_tracers(
                    self._fv3gfs, self._state
                ),
                self._increment_reservoir,
                self.monitor("dynamics", self._step_dynamics),
                self._step_prephysics,
                self._step_pre_radiation_physics,
                self._step_radiation_physics,
                self._step_post_radiation_physics,
                self.monitor(
                    "applied_physics",
                    self.emulate_or_prescribe_tendency(
                        self.monitor("fv3_physics", self._apply_physics)
                    ),
                ),
                self._compute_postphysics,
                self.monitor("python", self._apply_postphysics_to_dycore_state),
                self._apply_reservoir_update_to_state,
                self._intermediate_restarts,
            ]:
                with self._timer.clock(substep.__name__):
                    diagnostics.update(substep())
            yield self._state.time, {str(k): v for k, v in diagnostics.items()}
