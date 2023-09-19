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
)

import fv3fit
from fv3fit._shared.halos import append_halos_using_mpi
from fv3fit.reservoir.adapters import ReservoirDatasetAdapter
from runtime.names import SST
from .prescriber import sst_update_from_reference
from .machine_learning import rename_dataset_members, NameDict


logger = logging.getLogger(__name__)


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
    """

    models: Mapping[int, str]
    synchronize_steps: int = 1
    reservoir_timestep: str = "3h"  # TODO: Could this be inferred?
    time_average_inputs: bool = False
    diagnostic_only: bool = False
    warm_start: bool = False
    rename_mapping: NameDict = dataclasses.field(default_factory=dict)


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
        synchronize_steps: int,
        state_machine: Optional[_FiniteStateMachine] = None,
        diagnostic_only: bool = False,
        input_averager: Optional[TimeAverageInputs] = None,
        rename_mapping: Optional[NameDict] = None,
        warm_start: bool = False,
    ):
        self.model = model
        self.synchronize_steps = synchronize_steps
        self.initial_time = init_time
        self.timestep = reservoir_timestep
        self.diagnostic = diagnostic_only
        self.input_averager = input_averager
        self.warm_start = warm_start

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
                for k in self.model.input_variables
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
                    "MPI not available or tile dimension does not exist in state fields"
                    " during reservoir increment update"
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

    def predict(self, inputs, state):
        """Called at the end of timeloop after time has ticked from t -> t+1"""

        self._state_machine(self._state_machine.PREDICT)
        result = self.model.predict(inputs)
        output_state = rename_dataset_members(result, self.rename_mapping)

        diags = rename_dataset_members(
            output_state, {k: f"{k}_rc_out" for k in output_state}
        )

        for k, v in output_state.items():
            v.attrs["units"] = state[k].attrs.get("units", "unknown")

        # no halo necessary for potential hybrid inputs
        # +1 to align with the necessary increment before any prediction
        if (
            self._state_machine.completed_increments <= self.synchronize_steps
            or self.diagnostic
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
                    for k in self.model.model.hybrid_variables
                }
            )
        else:
            inputs = xr.Dataset()

        if self.input_averager is not None:
            self.input_averager.increment_running_average(inputs)

        if self._is_rc_update_step(time):
            logger.info(f"Reservoir model predict at time {time}")
            if self.input_averager is not None:
                inputs.update(self.input_averager.get_averages())
            tendencies, diags, state = self.predict(inputs, state)
            hybrid_diags = rename_dataset_members(
                inputs, {k: f"{self.rename_mapping.get(k, k)}_hyb_in" for k in inputs}
            )
            diags.update(hybrid_diags)
        else:
            tendencies, diags, state = {}, {}, {}

        return tendencies, diags, state


def open_rc_model(path: str) -> ReservoirDatasetAdapter:
    return cast(ReservoirDatasetAdapter, fv3fit.load(path))


def _get_time_averagers(model, do_time_average):
    if do_time_average:
        increment_averager = TimeAverageInputs(model.model.input_variables)
        predict_averager: Optional[TimeAverageInputs]
        if model.is_hybrid:
            hybrid_inputs = model.model.hybrid_variables
            variables = hybrid_inputs if hybrid_inputs is not None else []
            predict_averager = TimeAverageInputs(variables)
        else:
            predict_averager = None
    else:
        increment_averager, predict_averager = None, None

    return increment_averager, predict_averager


def get_reservoir_steppers(
    config: ReservoirConfig, rank: int, init_time: cftime.DatetimeJulian,
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

    incrementer = ReservoirIncrementOnlyStepper(
        model,
        init_time,
        rc_tdelta,
        config.synchronize_steps,
        state_machine=state_machine,
        input_averager=increment_averager,
        rename_mapping=config.rename_mapping,
        warm_start=config.warm_start,
    )
    predictor = ReservoirPredictStepper(
        model,
        init_time,
        rc_tdelta,
        config.synchronize_steps,
        state_machine=state_machine,
        diagnostic_only=config.diagnostic_only,
        input_averager=predict_averager,
        rename_mapping=config.rename_mapping,
        warm_start=config.warm_start,
    )
    return incrementer, predictor
