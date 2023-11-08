import cftime
import dataclasses
import logging
import pandas as pd
import xarray as xr
from datetime import timedelta
from typing import (
    Optional,
    MutableMapping,
    Hashable,
    Mapping,
    cast,
    Sequence,
    Dict,
    Union,
)

import fv3fit
from fv3fit._shared.halos import append_halos_using_mpi
from fv3fit.reservoir.adapters import ReservoirDatasetAdapter
import vcm
from runtime.names import SST, SPHUM, TEMP, PHYSICS_PRECIP_RATE, TOTAL_PRECIP
from runtime.tendency import add_tendency, tendencies_from_state_updates
from runtime.diagnostics import (
    enforce_heating_and_moistening_tendency_constraints,
    compute_diagnostics,
)
from .prescriber import sst_update_from_reference
from .machine_learning import rename_dataset_members, NameDict


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class TaperConfig:
    cutoff: int
    rate: float
    taper_dim: str = "z"

    def blend(self, prediction: xr.DataArray, input: xr.DataArray) -> xr.DataArray:
        n_levels = len(prediction[self.taper_dim])
        prediction_scaling = xr.DataArray(
            vcm.vertical_tapering_scale_factors(
                n_levels=n_levels, cutoff=self.cutoff, rate=self.rate
            ),
            dims=[self.taper_dim],
        )
        input_scaling = 1 - prediction_scaling

        return input_scaling * input + prediction_scaling * prediction


@dataclasses.dataclass
class ReservoirConfig:
    """
    Reservoir model configuration.

    Attributes:
        models: Mapping from rank to reservoir model path to load
        synchronize_steps: Number of steps to synchronize the reservoir
            before prediction
        reservoir_timestep: Timestep of the reservoir model
        time_average_inputs: Whether to time average inputs to the reservoir
            increment and prediction (if hybrid) steps.  Uses running averages
            to match the reservoir timestep.
        diagnostic_only: Whether to run the reservoir in diagnostic mode (no
            state updates)
        warm_start: Whether to use the saved state from a pre-synced reservoir
        rename_mapping: mapping from field names used in the underlying
            reservoir model to names used in fv3gfs wrapper
        hydrostatic (optional): whether simulation is hydrostatic.
            For net heating diagnostic. Defaults to false.
        mse_conserving_limiter (optional): whether to use MSE-conserving humidity
            limiter. Defaults to false.
    """

    models: Mapping[Union[int, str], str]
    synchronize_steps: int = 1
    reservoir_timestep: str = "3h"  # TODO: Could this be inferred?
    time_average_inputs: bool = False
    diagnostic_only: bool = False
    warm_start: bool = False
    rename_mapping: NameDict = dataclasses.field(default_factory=dict)
    hydrostatic: bool = False
    mse_conserving_limiter: bool = False
    interval_average_precipitation: bool = False
    taper_blending: Optional[Mapping] = None

    def __post_init__(self):
        # This handles cases in automatic config writing where json/yaml
        # do not allow integer keys
        _models = {}
        for key, url in self.models.items():
            try:
                int_key = int(key)
                _models[int_key] = url
            except (ValueError) as e:
                raise ValueError(
                    "Keys in reservoir_corrector.models must be integers "
                    "or string representation of integers."
                ) from e
        self.models = _models


class _FiniteStateMachine:
    """
    A simple state machine to keep to a shared state between the increment
    and predict steppers that are separated across the time loop.
    """

    INCREMENT = "increment"
    PREDICT = "predict"

    def __init__(self) -> None:
        self._last_called = None
        self._num_increments_completed = 0

    @property
    def completed_increments(self):
        return self._num_increments_completed

    def to_incremented(self):
        # incrementing allowed anytime, e.g., synchronizing
        self._last_called = self.INCREMENT
        self._num_increments_completed += 1

    def to_predicted(self):
        # predict only allowed after increment has been called
        if self._last_called != self.INCREMENT:
            raise ValueError("Must call increment before next prediction")
        self._last_called = self.PREDICT

    def __call__(self, state: str):
        if state == self.INCREMENT:
            self.to_incremented()
        elif state == self.PREDICT:
            self.to_predicted()
        else:
            raise ValueError(
                f"Unknown state provided to _ReservoirStepperState {state}"
            )


class PrecipTracker:
    def __init__(self, reservoir_timestep_seconds: float):
        self.reservoir_timestep_seconds = reservoir_timestep_seconds
        self.physics_precip_averager = TimeAverageInputs([PHYSICS_PRECIP_RATE])
        self._air_temperature_at_previous_interval = None
        self._specific_humidity_at_previous_interval = None

    def increment_physics_precip_rate(self, physics_precip_rate):
        self.physics_precip_averager.increment_running_average(
            {PHYSICS_PRECIP_RATE: physics_precip_rate}
        )

    def interval_avg_precip_rates(self, net_moistening_due_to_reservoir):
        physics_precip_rate = self.physics_precip_averager.get_averages()[
            PHYSICS_PRECIP_RATE
        ]
        total_precip_rate = physics_precip_rate - net_moistening_due_to_reservoir
        total_precip_rate = total_precip_rate.where(total_precip_rate >= 0, 0)
        reservoir_precip_rate = total_precip_rate - physics_precip_rate
        return {
            "total_precip_rate_res_interval_avg": total_precip_rate,
            "physics_precip_rate_res_interval_avg": physics_precip_rate,
            "reservoir_precip_rate_res_interval_avg": reservoir_precip_rate,
        }

    def accumulated_precip_update(
        self,
        physics_precip_total_over_model_timestep,
        reservoir_precip_rate_over_res_interval,
        reservoir_timestep,
    ):
        # Since the reservoir correction is only applied every reservoir_timestep,
        # all of the precip due to the reservoir is put into the accumulated precip
        # in the model timestep at update time.
        m_per_mm = 1 / 1000
        reservoir_total_precip = (
            reservoir_precip_rate_over_res_interval * reservoir_timestep * m_per_mm
        )
        total_precip = physics_precip_total_over_model_timestep + reservoir_total_precip
        total_precip.attrs["units"] = "m"
        return total_precip


class TimeAverageInputs:
    """
    Copy of time averaging components from runtime.diagnostics.manager to
    use for averaging inputs to the reservoir model.
    """

    def __init__(self, variables: Sequence[str]):
        self.variables = variables
        self._running_total: Dict[str, xr.DataArray] = {}
        self._n = 0
        self._recorded_units: Dict[str, str] = {}

    def increment_running_average(self, inputs: Mapping[str, xr.DataArray]):
        for key in inputs:
            self._recorded_units[key] = inputs[key].attrs.get("units", "unknown")

        for key in self.variables:
            if key in self._running_total:
                self._running_total[key] += inputs[key]
            else:
                self._running_total[key] = inputs[key].copy()

        self._n += 1

    def _reset_running_average(self):
        self._running_total = {}
        self._n = 0

    def get_averages(self):
        if not self._running_total and self.variables:
            raise ValueError(
                f"Average called when no fields ({self.variables})"
                " present in running average."
            )

        averaged_data = {key: val / self._n for key, val in self._running_total.items()}
        for key in averaged_data:
            averaged_data[key].attrs["units"] = self._recorded_units[key]

        self._reset_running_average()
        logger.info(
            "Retrieved time averaged input data for reservoir:"
            f" {averaged_data.keys()}"
        )

        return averaged_data


class _ReservoirStepper:

    label = "base_reservoir_stepper"

    def __init__(
        self,
        model: ReservoirDatasetAdapter,
        init_time: cftime.DatetimeJulian,
        reservoir_timestep: timedelta,
        model_timestep: float,
        synchronize_steps: int,
        state_machine: Optional[_FiniteStateMachine] = None,
        diagnostic_only: bool = False,
        input_averager: Optional[TimeAverageInputs] = None,
        rename_mapping: Optional[NameDict] = None,
        warm_start: bool = False,
        hydrostatic: bool = False,
        mse_conserving_limiter: bool = False,
        precip_tracker: Optional[PrecipTracker] = None,
        taper_blending: Optional[TaperConfig] = None,
    ):
        self.model = model
        self.synchronize_steps = synchronize_steps
        self.initial_time = init_time
        self.timestep = reservoir_timestep
        self.model_timestep = model_timestep
        self.is_diagnostic = diagnostic_only
        self.input_averager = input_averager
        self.warm_start = warm_start
        self.hydrostatic = hydrostatic
        self.mse_conserving_limiter = mse_conserving_limiter
        self.precip_tracker = precip_tracker
        self.taper_blending = taper_blending

        if state_machine is None:
            state_machine = _FiniteStateMachine()
        self._state_machine = state_machine

        if self.warm_start:
            if self.synchronize_steps != 0:
                raise ValueError(
                    "Warm start specified with non-zero sync steps.  Ensure that"
                    " the reservoir model is pre-synchronized and set sync steps to 0"
                    " in the configuration."
                )

            # allows for immediate predict
            self._state_machine(self._state_machine.INCREMENT)

        if rename_mapping is None:
            rename_mapping = cast(NameDict, {})
        self.rename_mapping = rename_mapping

    @property
    def completed_sync_steps(self):
        return self._state_machine.completed_increments

    def __call__(self, time, state):
        raise NotImplementedError(
            "Must use a wrapper Stepper for the reservoir to use in the TimeLoop"
        )

    def _is_rc_update_step(self, time):
        remainder = (time - self.initial_time) % self.timestep
        return remainder == timedelta(0)

    def get_diagnostics(self, state, tendency):
        diags: MutableMapping[Hashable, xr.DataArray] = {}
        return diags, xr.DataArray()


class ReservoirIncrementOnlyStepper(_ReservoirStepper):
    """
    Stepper that only increments the state of the reservoir.  Useful because we
    need to call this using the input of the model prior to any updates from other
    steppers.  The model adapter should be the same adapter provided to the
    ReservoirStepper.
    """

    label = "reservoir_incrementer"

    def _get_inputs_from_state(self, state):
        """
        Get all required inputs for incrementing w/ halos

        Add the slmask if SST is an input variable for masking
        """

        reservoir_inputs = xr.Dataset(
            {
                k: state[self.rename_mapping.get(k, k)]
                for k in self.model.nonhybrid_input_variables
            }
        )
        n_halo_points = self.model.input_overlap
        if n_halo_points > 0:
            try:
                rc_in_with_halos = append_halos_using_mpi(
                    reservoir_inputs, n_halo_points
                )
            except RuntimeError:
                raise ValueError(
                    "MPI not available or tile dimension does not exist in state "
                    "fields during reservoir increment update"
                )
            reservoir_inputs = rc_in_with_halos

        return reservoir_inputs

    def increment_reservoir(self, inputs):
        """Should be called at beginning of time loop"""

        if self.completed_sync_steps == 0 and not self.warm_start:
            self.model.reset_state()
        self._state_machine(self._state_machine.INCREMENT)
        self.model.increment_state(inputs)

    def __call__(self, time, state):

        diags = {}

        # add to averages
        inputs = self._get_inputs_from_state(state)
        if self.input_averager is not None:
            self.input_averager.increment_running_average(inputs)

        if self._is_rc_update_step(time):
            if self.input_averager is not None:
                # update inputs w/ average quantities
                inputs.update(self.input_averager.get_averages())

            logger.info(f"Incrementing rc at time {time}")
            self.increment_reservoir(inputs)
            diags = rename_dataset_members(
                inputs, {k: f"{self.rename_mapping.get(k, k)}_rc_in" for k in inputs}
            )

            # prevent conflict with non-halo diagnostics
            if self.model.input_overlap > 0:
                overlap = self.model.input_overlap
                isel_kwargs = {
                    dim: slice(overlap, -overlap)
                    for dim in diags.dims
                    if dim in ["x", "y"]
                }
                diags.isel(**isel_kwargs)

        return {}, diags, {}


class ReservoirPredictStepper(_ReservoirStepper):
    """
    Stepper that predicts using the current state of the reservoir.  Meant to
    be called after ReservoirIncrementOnlyStepper has been called but it's left
    up to the caller to ensure these occur in the right order.
    """

    label = "reservoir_predictor"
    DIAGS_OUTPUT_SUFFIX = "rc_out"

    def predict(self, inputs, state):
        """Called at the end of timeloop after time has ticked from t -> t+1"""

        self._state_machine(self._state_machine.PREDICT)
        result = self.model.predict(inputs)

        output_state = rename_dataset_members(result, self.rename_mapping)

        diags = rename_dataset_members(
            output_state, {k: f"{k}_{self.DIAGS_OUTPUT_SUFFIX}" for k in output_state}
        )
        if self.taper_blending is not None:
            input_renaming = {
                k: v for k, v in self.rename_mapping.items() if k in inputs
            }
            output_state = self.taper_blending.blend(
                output_state, inputs.rename(input_renaming)
            )
        for k, v in output_state.items():
            v.attrs["units"] = state[k].attrs.get("units", "unknown")

        # no halo necessary for potential hybrid inputs
        # +1 to align with the necessary increment before any prediction
        if (
            self._state_machine.completed_increments <= self.synchronize_steps
            or self.is_diagnostic
        ):
            output_state = {}

        if SST in output_state:
            sst_updates = sst_update_from_reference(
                state, output_state, reference_sst_name=SST
            )
            output_state.update(sst_updates)

        return {}, diags, output_state

    def __call__(self, time, state):

        # won't evaluate to true until we've reached the step before the next increment
        # e.g., if fv3 has k timesteps between rc timestep, on t + k - 1, the timestep
        # at the end will have ticked over to t + k in the middle of the called wrapper
        # steps prior to predict, we'll maybe use the integrated
        # hybrid quantites from t -> t + k, make the rc prediction for t + k, and then
        # increment during the next time loop based on those outputs.

        if self.model.is_hybrid:
            inputs = xr.Dataset(
                {
                    k: state[self.rename_mapping.get(k, k)]
                    for k in self.model.hybrid_variables
                }
            )
        else:
            inputs = xr.Dataset()

        if self.input_averager is not None:
            self.input_averager.increment_running_average(inputs)

        if self.precip_tracker is not None:
            self.precip_tracker.increment_physics_precip_rate(
                state[PHYSICS_PRECIP_RATE]
            )

        if self._is_rc_update_step(time):
            logger.info(f"Reservoir model predict at time {time}")
            if self.input_averager is not None:
                inputs.update(self.input_averager.get_averages())

            tendencies, diags, updated_state = self.predict(inputs, state)

            hybrid_diags = rename_dataset_members(
                inputs, {k: f"{self.rename_mapping.get(k, k)}_hyb_in" for k in inputs}
            )
            diags.update(hybrid_diags)

            # This check is done on the _rc_out diags since those are always available.
            # This allows zero field diags to be returned on timesteps where the
            # reservoir is not updating the state.
            diags_Tq_vars = {f"{v}_{self.DIAGS_OUTPUT_SUFFIX}" for v in [TEMP, SPHUM]}

            if diags_Tq_vars.issubset(list(diags.keys())):
                # TODO: Currently the reservoir only predicts updated states and returns
                # empty tendencies. If tendency predictions are implemented in the
                # prognostic run, the limiter/conservation updates should be updated to
                # take this option into account and use predicted tendencies directly.
                tendencies_from_state_prediction = tendencies_from_state_updates(
                    initial_state=state,
                    updated_state=updated_state,
                    dt=self.model_timestep,
                )
                (
                    tendency_updates_from_constraints,
                    diagnostics_updates_from_constraints,
                ) = enforce_heating_and_moistening_tendency_constraints(
                    state=state,
                    tendency=tendencies_from_state_prediction,
                    timestep=self.model_timestep,
                    mse_conserving=self.mse_conserving_limiter,
                    hydrostatic=self.hydrostatic,
                    temperature_tendency_name="dQ1",
                    humidity_tendency_name="dQ2",
                    zero_fill_missing_tendencies=True,
                )

                diags.update(diagnostics_updates_from_constraints)
                updated_state = add_tendency(
                    state=state,
                    tendencies=tendency_updates_from_constraints,
                    dt=self.model_timestep,
                )
                # Adjust corrective tendencies to be averages over
                # the full reservoir timestep
                for key in tendency_updates_from_constraints:
                    if key != "specific_humidity_limiter_active":
                        tendency_updates_from_constraints[key] *= (
                            self.model_timestep / self.timestep.total_seconds()
                        )
                tendencies.update(tendency_updates_from_constraints)

        else:
            tendencies, diags, updated_state = {}, {}, {}

        return tendencies, diags, updated_state

    def get_diagnostics(self, state, tendency):
        diags = compute_diagnostics(state, tendency, self.label, self.hydrostatic)
        return diags, diags[f"net_moistening_due_to_{self.label}"]

    def update_precip(
        self, physics_precip, net_moistening_due_to_reservoir,
    ):
        diags = {}

        # running average gets reset in this call
        precip_rates = self.precip_tracker.interval_avg_precip_rates(
            net_moistening_due_to_reservoir
        )
        diags.update(precip_rates)

        diags[TOTAL_PRECIP] = self.precip_tracker.accumulated_precip_update(
            physics_precip,
            diags["reservoir_precip_rate_res_interval_avg"],
            self.timestep.total_seconds(),
        )
        return diags


def open_rc_model(path: str) -> ReservoirDatasetAdapter:
    return cast(ReservoirDatasetAdapter, fv3fit.load(path))


def _get_time_averagers(model, do_time_average):
    if do_time_average:
        increment_averager = TimeAverageInputs(model.model.input_variables)
        predict_averager: Optional[TimeAverageInputs]
        if model.is_hybrid:
            hybrid_inputs = model.hybrid_variables
            variables = hybrid_inputs if hybrid_inputs is not None else []
            predict_averager = TimeAverageInputs(variables)
        else:
            predict_averager = None
    else:
        increment_averager, predict_averager = None, None

    return increment_averager, predict_averager


def get_reservoir_steppers(
    config: ReservoirConfig,
    rank: int,
    init_time: cftime.DatetimeJulian,
    model_timestep: float,
):
    """
    Gets both steppers needed by the time loop to increment the state using
    inputs from the beginning of the timestep and applying hybrid readout
    using the stepped underlying model + incremented RC state.
    """
    try:
        model = open_rc_model(config.models[rank])
    except KeyError:
        raise KeyError(
            f"No reservoir model path found  for rank {rank}. "
            "Ensure that the rank key and model is present in the configuration."
        )
    state_machine = _FiniteStateMachine()
    rc_tdelta = pd.to_timedelta(config.reservoir_timestep)
    increment_averager, predict_averager = _get_time_averagers(
        model, config.time_average_inputs
    )
    _precip_tracker_kwargs = {}
    if config.interval_average_precipitation:
        _precip_tracker_kwargs["precip_tracker"] = PrecipTracker(
            reservoir_timestep_seconds=rc_tdelta.total_seconds(),
        )

    if config.taper_blending is not None:
        if len({"cutoff", "rate"}.intersection(config.taper_blending.keys())) == 2:
            taper_blending = TaperConfig(**config.taper_blending)

    incrementer = ReservoirIncrementOnlyStepper(
        model,
        init_time,
        reservoir_timestep=rc_tdelta,
        synchronize_steps=config.synchronize_steps,
        state_machine=state_machine,
        input_averager=increment_averager,
        rename_mapping=config.rename_mapping,
        warm_start=config.warm_start,
        model_timestep=model_timestep,
    )
    predictor = ReservoirPredictStepper(
        model,
        init_time,
        reservoir_timestep=rc_tdelta,
        synchronize_steps=config.synchronize_steps,
        state_machine=state_machine,
        diagnostic_only=config.diagnostic_only,
        input_averager=predict_averager,
        rename_mapping=config.rename_mapping,
        warm_start=config.warm_start,
        model_timestep=model_timestep,
        hydrostatic=config.hydrostatic,
        mse_conserving_limiter=config.mse_conserving_limiter,
        taper_blending=taper_blending,
        **_precip_tracker_kwargs,
    )
    return incrementer, predictor
