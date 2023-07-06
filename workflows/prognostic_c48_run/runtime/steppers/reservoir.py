import cftime
import dataclasses
from datetime import timedelta
import pandas as pd
from typing import Optional, MutableMapping, Hashable
import xarray as xr

from fv3fit._shared.halos import append_halos_using_mpi
from fv3fit.reservoir.model import HybridReservoirComputingModel, HybridDatasetAdapter


@dataclasses.dataclass
class HybridReservoirConfig:
    """
    Hybrid reservoir configuration.

    Attributes:
        model: URL to the global hybrid RC model
    """

    model: str
    synchronize_steps: int = 1  # TODO: this could also be set as a duration
    reservoir_timestep: str = "3h"  # TODO: could this be inferred from the model?


class _ReservoirStepperState:

    INCREMENT = "increment"
    PREDICT = "predict"

    def __init__(self) -> None:
        self._last_called = None

    def increment(self):
        # incrementing allowed anytime, e.g., synchronizing
        self._last_called = self.INCREMENT

    def predict(self):
        # predict only allowed after increment has been called
        if self._last_called != self.INCREMENT:
            raise ValueError("Must call increment before next prediction")
        self._last_called = self.PREDICT

    def __call__(self, state: str):
        if state == self.INCREMENT:
            self.increment()
        elif state == self.PREDICT:
            self.predict()
        else:
            raise ValueError(
                f"Unknown state provided to _ReservoirStepperState {state}"
            )


class _ReservoirStepper:

    label = "hybrid_reservoir"

    def __init__(
        self,
        model: HybridDatasetAdapter,
        reservoir_timestep: timedelta,
        synchronize_steps: int,
        model_timestep_seconds: int = 900,
        state_checker: Optional[_ReservoirStepperState] = None,
    ):
        self.model = model
        self.rc_timestep = reservoir_timestep
        self.synchronize_steps = synchronize_steps
        self.dt_atmos = timedelta(seconds=model_timestep_seconds)
        self.init_time: Optional[cftime.DatetimeJulian] = None
        self.completed_sync_steps = 0
        self._state_machine = state_checker

    def increment_reservoir(self, time, state):
        """Should be called at beginning of time loop"""

        if self.init_time is None:
            self.init_time = time

        if self._is_rc_update_step(time):
            reservoir_inputs = state[self.model.input_variables]

            try:
                n_halo_points = self.model.rank_divider.overlap
                rc_in_with_halos = append_halos_using_mpi(
                    reservoir_inputs, n_halo_points
                )
            except RuntimeError:
                raise ValueError(
                    "MPI not available or tile dimension does not exist in state fields"
                    " during reservoir increment update"
                )

            if self.completed_sync_steps == 0:
                self.model.reset_state()

            self.model.increment_state(rc_in_with_halos)
            self.completed_sync_steps += 1

            if self._state_machine is not None:
                self._state_machine(self._state_machine.INCREMENT)

        return {}, {}, {}

    def hybrid_predict(self, time, state):

        if self._is_rc_update_step(time):
            hybrid_inputs = state[self.model.hybrid_variables]

            try:
                n_halo_points = self.model.rank_divider.overlap
                hybrid_in_with_halos = append_halos_using_mpi(
                    hybrid_inputs, n_halo_points
                )
            except RuntimeError:
                raise ValueError(
                    "MPI not available or tile dimension does not exist in state fields"
                    " during RC stepper update"
                )

            updated_state = {}
            if self.completed_sync_steps >= self.synchronize_steps:
                result = self.model.predict(hybrid_in_with_halos)
                updated_state.update(
                    {k: result[k] for k in self.model.output_variables}
                )

            if self._state_machine is not None:
                self._state_machine(self._state_machine.PREDICT)

        return {}, {}, updated_state

    def __call__(self, time, state):
        raise NotImplementedError(
            "Must use a wrapper Stepper for the reservoir to use in the TimeLoop"
        )

    def _is_rc_update_step(self, time):
        return (time - self.init_time) % self.rc_timestep == 0

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

    label = "hybrid_reservoir_incrementer"

    def __call__(self, time, state):
        return super().increment_reservoir(time, state)


class ReservoirHybridPredictStepper(_ReservoirStepper):
    """
    Stepper that predicts using the current state of the reservoir.  Meant to
    be called after ReservoirIncrementOnlyStepper has been called but it's left
    up to the caller to ensure these occur in the right order.
    """

    label = "hybrid_reservoir_incrementer"

    def __call__(self, time, state):
        return super().hybrid_predict(time, state)


def open_rc_model(path: str):
    return HybridDatasetAdapter(HybridReservoirComputingModel.load(path))


def get_reservoir_steppers(config: HybridReservoirConfig):
    """
    Gets both steppers needed by the time loop to increment the state using
    inputs from the beginning of the timestep and applying hybrid readout
    using the stepped underlying model + incremented RC state.
    """

    model = open_rc_model(config.model)
    state_machine = _ReservoirStepperState()
    rc_tdelta = pd.to_timedelta(config.reservoir_timestep)
    incrementer = ReservoirIncrementOnlyStepper(
        model, rc_tdelta, config.synchronize_steps, state_checker=state_machine
    )
    predictor = ReservoirHybridPredictStepper(
        model, rc_tdelta, config.synchronize_steps, state_checker=state_machine
    )
    return incrementer, predictor
