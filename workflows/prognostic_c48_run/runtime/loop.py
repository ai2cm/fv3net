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
import runtime.diagnostics.tracers
from runtime.monitor import Monitor
from runtime.names import (
    TENDENCY_TO_STATE_NAME,
    TOTAL_PRECIP_RATE,
    PREPHYSICS_OVERRIDES,
    WIND_TENDENCIES,
)
from runtime.steppers.machine_learning import (
    MachineLearningConfig,
    PureMLStepper,
    download_model,
    open_model,
)
from runtime.steppers.nudging import PureNudger
from runtime.steppers.prescriber import Prescriber, PrescriberConfig
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


def fillna_tendency(tendency: xr.DataArray) -> Tuple[xr.DataArray, xr.DataArray]:
    tendency_filled = tendency.fillna(0.0)
    tendency_filled_frac = (
        xr.where(tendency != tendency_filled, 1, 0).sum("z") / tendency.sizes["z"]
    )
    return tendency_filled, tendency_filled_frac


def transform_agrid_winds_to_dgrid_winds(
    ua: xr.DataArray, va: xr.DataArray
) -> Tuple[xr.DataArray, xr.DataArray]:
    ua_quantity = pace.util.Quantity.from_data_array(ua)
    va_quantity = pace.util.Quantity.from_data_array(va)
    x_wind, y_wind = fv3gfs.wrapper.transform_agrid_winds_dgrid_winds(
        ua_quantity, va_quantity
    )
    return x_wind.data_array, y_wind.data_array


def compute_total_dgrid_wind_tendencies(state, tendencies):
    # TODO: this could be computationally optimized.  If no A-grid wind
    # tendencies are provided, we don't need to bother passing zeros through
    # to the transformation function.  We don't need to bother adding zeros
    # if no D-grid wind tendencies are provided either.
    dQx_wind = tendencies.get(
        "dQx_wind", xr.zeros_like(state["x_wind"]).assign_attrs(units="m/s/s")
    )
    dQy_wind = tendencies.get(
        "dQy_wind", xr.zeros_like(state["y_wind"]).assign_attrs(units="m/s/s")
    )
    dQu = tendencies.get(
        "dQu", xr.zeros_like(state["eastward_wind"]).assign_attrs(units="m/s/s")
    )
    dQv = tendencies.get(
        "dQv", xr.zeros_like(state["northward_wind"]).assign_attrs(units="m/s/s")
    )
    (
        dQx_wind_from_dQu_and_dQv,
        dQy_wind_from_dQu_and_dQv,
    ) = transform_agrid_winds_to_dgrid_winds(dQu, dQv)
    return dQx_wind + dQx_wind_from_dQu_and_dQv, dQy_wind + dQy_wind_from_dQu_and_dQv


def add_tendency(state: Any, tendencies: State, dt: float) -> Tuple[State, State]:
    """Given state and tendency prediction, return updated state, which only includes
    variables updated by tendencies. Also returns column-integrated fraction of
    tendencies which are filled nans.
    """

    with xr.set_options(keep_attrs=True):
        updated = {}
        filled_tendencies = {}
        filled_fractions = {}

        for name, tendency in tendencies.items():
            filled, filled_fraction = fillna_tendency(tendency[name])
            filled_tendencies[name] = filled
            filled_fractions[f"{name}_filled_frac"] = filled_fraction

        if set(filled_tendencies).intersection(WIND_TENDENCIES):
            # TODO: it's an open question how best to handle diagnostics in this
            # case.  Right now I am permitting specifying tendency updates to
            # both the D-grid winds and the A-grid winds at the same time, and
            # am taking their sum to apply to the D-grid winds in the end.  The
            # filled tendencies remain separate, however.  In other words if
            # both x_wind and eastward_wind tendencies are provided, the
            # diagnostic for the x_wind tendency is just what was originally
            # provided on the D-grid (not the sum of the D-grid and A-grid
            # contributions).  I believe this is the way things would have
            # been handled prior to this change as well.
            dQx_wind_total, dQy_wind_total = compute_total_dgrid_wind_tendencies(
                state, filled_tendencies
            )
            updated["x_wind"] = state["x_wind"] + dQx_wind_total * dt
            updated["y_wind"] = state["y_wind"] + dQy_wind_total * dt

        for name, tendency in filled_tendencies.items():
            if name not in WIND_TENDENCIES:
                try:
                    state_name = str(TENDENCY_TO_STATE_NAME[name])
                except KeyError:
                    raise KeyError(
                        f"Tendency variable '{name}' does not have an entry mapping it "
                        "to a corresponding state variable to add to. "
                        "Existing tendencies with mappings to state are "
                        f"{list(TENDENCY_TO_STATE_NAME.keys())}"
                    )
                updated[state_name] = state[state_name] + tendency * dt

    return updated, filled_fractions  # type: ignore


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
        self._timer = pace.util.Timer()
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
        self._radiation_stepper = self._get_radiation_stepper(config, namelist)
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

    def _get_prescriber_or_ml_stepper(
        self,
        stepper_config: Union[PrescriberConfig, MachineLearningConfig],
        step: str,
        hydrostatic: bool = False,
    ) -> Union[PureMLStepper, Prescriber]:
        if isinstance(stepper_config, MachineLearningConfig):
            model = self._open_model(stepper_config)
            stepper: Union[PureMLStepper, Prescriber] = PureMLStepper(
                model, self._timestep, hydrostatic
            )
            self._log_info(f"Using PureMLStepper at {step}.")
        else:
            stepper = runtime.factories.get_prescriber(
                stepper_config, self._get_communicator()
            )
            self._log_info(
                "Using Prescriber for variables "
                f"{list(stepper_config.variables.values())} at {step}."
            )
        return stepper

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
                prephysics_steppers.append(
                    self._get_prescriber_or_ml_stepper(prephysics_config, "prephysics")
                )
            stepper = CombinedStepper(prephysics_steppers)
        return stepper

    def _get_postphysics_stepper(
        self, config: UserConfig, hydrostatic: bool
    ) -> Optional[Stepper]:
        if config.scikit_learn:
            self._log_info("Using MLStepper for postphysics updates")
            if config.scikit_learn.use_mse_conserving_humidity_limiter:
                self._log_info("Using MSE-conserving moisture limiter for postphysics")
            else:
                self._log_info(
                    "Using old non-MSE-conserving moisture limiter for postphysics"
                )
            model = self._open_model(config.scikit_learn)
            stepper: Optional[Stepper] = PureMLStepper(
                model,
                self._timestep,
                hydrostatic,
                config.scikit_learn.use_mse_conserving_humidity_limiter,
            )
        elif config.nudging:
            self._log_info("Using NudgingStepper for postphysics updates")
            stepper = PureNudger(config.nudging, self._get_communicator(), hydrostatic)
        else:
            self._log_info("Performing baseline simulation")
            stepper = None
        return stepper

    def _get_radiation_stepper(
        self, config: UserConfig, namelist: Mapping[Hashable, Any]
    ) -> Optional[Stepper]:
        if config.radiation_scheme is not None:
            radiation_input_generator_config = config.radiation_scheme.input_generator
            if radiation_input_generator_config is not None:
                radiation_input_generator: Optional[
                    Union[PureMLStepper, Prescriber]
                ] = self._get_prescriber_or_ml_stepper(
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
        pass
        # self._log_debug(f"Apply postphysics tendencies to physics state")
        # tendency = {k: v for k, v in self._tendencies.items() if k in ["dQu", "dQv"]}

        # diagnostics: Diagnostics = {}

        # if self._postphysics_stepper is not None:
        #     diagnostics = self._postphysics_stepper.get_momentum_diagnostics(
        #         self._state, tendency
        #     )
        #     if self._postphysics_only_diagnostic_ml:
        #         rename_diagnostics(diagnostics)
        #     else:
        #         updated_state, tendency_filled_frac = add_tendency(
        #             self._state, tendency, dt=self._timestep
        #         )
        #         self._state.update_mass_conserving(updated_state)
        #         diagnostics.update(tendency_filled_frac)

        # return diagnostics

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
                updated_state_from_tendency, tendency_filled_frac = add_tendency(
                    self._state, self._tendencies, dt=self._timestep
                )

                # if total precip is updated directly by stepper,
                # it will overwrite this precipitation_sum
                updated_state_from_tendency[TOTAL_PRECIP] = precipitation_sum(
                    self._state[TOTAL_PRECIP], net_moistening, self._timestep,
                )
                self._state.update_mass_conserving(updated_state_from_tendency)
                diagnostics.update(tendency_filled_frac)
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

    def _intermediate_restarts(self) -> Diagnostics:
        self._log_info("Saving intermediate restarts if enabled.")
        self._fv3gfs.save_intermediate_restart_if_enabled()
        return {}

    def __iter__(
        self,
    ) -> Iterator[Tuple[cftime.DatetimeJulian, Dict[str, xr.DataArray]]]:

        for i in range(self._fv3gfs.get_step_count()):
            diagnostics: Diagnostics = {}
            for substep in [
                lambda: runtime.diagnostics.tracers.compute_column_integrated_tracers(
                    self._state
                ),
                self.monitor("dynamics", self._step_dynamics),
                self._step_prephysics,
                self._step_pre_radiation_physics,
                self._step_radiation_physics,
                self._step_post_radiation_physics,
                self._apply_postphysics_to_physics_state,
                self.monitor(
                    "applied_physics",
                    self.emulate_or_prescribe_tendency(
                        self.monitor("fv3_physics", self._apply_physics)
                    ),
                ),
                self._compute_postphysics,
                self.monitor("python", self._apply_postphysics_to_dycore_state),
                self._intermediate_restarts,
            ]:
                with self._timer.clock(substep.__name__):
                    diagnostics.update(substep())
            yield self._state.time, {str(k): v for k, v in diagnostics.items()}
