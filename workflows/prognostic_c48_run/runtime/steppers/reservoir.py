import cftime
import dataclasses
from datetime import timedelta
import pandas as pd
from typing import Optional, MutableMapping, Hashable, Sequence, cast
import xarray as xr

import fv3fit
from fv3fit._shared import put_dir
from fv3fit._shared.halos import append_halos_using_mpi
from fv3fit.reservoir.adapters import ReservoirDatasetAdapter


@dataclasses.dataclass
class ReservoirConfig:
    """
    Reservoir model configuration.

    Attributes:
        model: URL to the global hybrid RC model
    """

    models: Sequence[str]
    synchronize_steps: int = 1
    reservoir_timestep: str = "3h"  # TODO: Could this be inferred?


class _FiniteStateMachine:
    """
    A simple state machine to keep to a shared state between the increment
    and predict steppers that are separated across the time loop.
    """

    INCREMENT = "increment"
    PREDICT = "predict"

    def __init__(self) -> None:
        self._last_called = None
        self._init_time = None
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


class _InitTimeStore:
    """
    A time store to keep a shared init time variable between the increment
    and predict steppers that are separated across the time loop.
    """

    def __init__(self):
        self._init_time = None

    @property
    def init_time(self):
        return self._init_time

    def set_init_time(self, time: cftime.DatetimeJulian):
        self._init_time = time


class _ReservoirStepper:

    label = "base_reservoir_stepper"

    def __init__(
        self,
        model: ReservoirDatasetAdapter,
        reservoir_timestep: timedelta,
        synchronize_steps: int,
        model_timestep_seconds: int = 900,
        state_machine: Optional[_FiniteStateMachine] = None,
        init_time_store: Optional[_InitTimeStore] = None,
    ):
        self.model = model
        self.rc_timestep = reservoir_timestep
        self.synchronize_steps = synchronize_steps
        self.dt_atmos = timedelta(seconds=model_timestep_seconds)
        self._init_time = None

        if state_machine is None:
            self._state_machine = _FiniteStateMachine()
        else:
            self._state_machine = state_machine

        if init_time_store is None:
            self._init_time_store = _InitTimeStore()
        else:
            self._init_time_store = init_time_store

    @property
    def init_time(self):
        return self._init_time_store.init_time

    @property
    def completed_sync_steps(self):
        return self._state_machine.completed_increments

    def __call__(self, time, state):
        raise NotImplementedError(
            "Must use a wrapper Stepper for the reservoir to use in the TimeLoop"
        )

    def _is_rc_update_step(self, time):
        if self.init_time is None:
            raise ValueError(
                "Cannot determine reservoir update status without init time.  Ensure"
                " that a _ReservoirStateStepper has an init time specified or that the"
                " reservoir increment stepper is called at least once."
            )
        return (time - self.init_time) % self.rc_timestep == timedelta(seconds=0)

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

    def increment_reservoir(self, state):
        """Should be called at beginning of time loop"""
        reservoir_inputs = state[self.model.input_variables]

        try:
            n_halo_points = self.model.rank_divider.overlap
            rc_in_with_halos = append_halos_using_mpi(reservoir_inputs, n_halo_points)
        except RuntimeError:
            raise ValueError(
                "MPI not available or tile dimension does not exist in state fields"
                " during reservoir increment update"
            )

        if self.completed_sync_steps == 0:
            self.model.reset_state()

        self._state_machine(self._state_machine.INCREMENT)
        self.model.increment_state(rc_in_with_halos)

    def __call__(self, time, state):

        if self.init_time is None:
            self._init_time_store.set_init_time(time)

        if self._is_rc_update_step(time):
            self.increment_reservoir(state)

        return {}, {}, {}


class ReservoirPredictStepper(_ReservoirStepper):
    """
    Stepper that predicts using the current state of the reservoir.  Meant to
    be called after ReservoirIncrementOnlyStepper has been called but it's left
    up to the caller to ensure these occur in the right order.
    """

    label = "reservoir_predictor"

    def predict(self, state):
        """Called at the end of timeloop after time has ticked from t -> t+1"""

        updated_state = {}
        diags = {}

        inputs = state[self.model.input_variables]
        self._state_machine(self._state_machine.PREDICT)

        # no halo necessary for potential hybrid inputs
        # +1 to align with the necessary increment before any prediction
        if self._state_machine.completed_increments >= self.synchronize_steps + 1:
            result = self.model.predict(inputs)
            updated_state.update({k: result[k] for k in self.model.output_variables})
            diags.update({f"{k}_rc_out": v for k, v in updated_state.items()})

        return {}, diags, updated_state

    def __call__(self, time, state):

        # won't evaluate to true until we've reached the step before the next increment
        # e.g., if fv3 has k timesteps between rc timestep, on t + k - 1, the timestep
        # at the end will have ticked over to t + k in the middle of the called wrapper
        # steps prior to predict, we'll maybe use the integrated
        # hybrid quantites from t -> t + k, make the rc prediction for t + k, and then
        # increment during the next time loop based on those outputs.
        if self._is_rc_update_step(time):
            tendencies, diags, state = self.predict(state)
        else:
            tendencies, diags, state = {}, {}, {}

        return tendencies, diags, state


def open_rc_model(path: str) -> ReservoirDatasetAdapter:
    with put_dir(path) as local_path:
        return cast(ReservoirDatasetAdapter, fv3fit.load(local_path))


def get_reservoir_steppers(config: ReservoirConfig, rank: int):
    """
    Gets both steppers needed by the time loop to increment the state using
    inputs from the beginning of the timestep and applying hybrid readout
    using the stepped underlying model + incremented RC state.
    """
    try:
        model = open_rc_model(config.models[rank])
    except IndexError:
        raise IndexError(
            "Not enough models provided in the stepper ReservoirConfig"
            " for number of MPI ranks"
        )
    state_machine = _FiniteStateMachine()
    rc_tdelta = pd.to_timedelta(config.reservoir_timestep)
    incrementer = ReservoirIncrementOnlyStepper(
        model, rc_tdelta, config.synchronize_steps, state_machine=state_machine
    )
    predictor = ReservoirPredictStepper(
        model, rc_tdelta, config.synchronize_steps, state_machine=state_machine
    )
    return incrementer, predictor
