import functools
from typing import Any

import fv3gfs.util
import fv3gfs.wrapper

from runtime.steppers.base import (
    Stepper,
    State,
    Diagnostics,
    apply,
    precipitation_sum,
    LoggingMixin,
)

from runtime.diagnostics.machine_learning import compute_nudging_diagnostics

from runtime.nudging import (
    nudging_timescales_from_dict,
    setup_get_reference_state,
    get_nudging_tendency,
    get_reference_surface_temperatures,
    NudgingConfig,
)

from runtime.names import TOTAL_PRECIP


SST_NAME = "ocean_surface_temperature"
TSFC_NAME = "surface_temperature"
MASK_NAME = "land_sea_mask"


class PureNudger:

    name = "nudging"

    def __init__(
        self, config: NudgingConfig, communicator: fv3gfs.util.CubedSphereCommunicator,
    ):

        variables_to_nudge = list(config.timescale_hours)
        self._get_reference_state = setup_get_reference_state(
            config,
            variables_to_nudge + [SST_NAME, TSFC_NAME],
            fv3gfs.wrapper.get_tracer_metadata(),
            communicator,
        )

        self._nudging_timescales = nudging_timescales_from_dict(config.timescale_hours)
        self._get_nudging_tendency = functools.partial(
            get_nudging_tendency, nudging_timescales=self._nudging_timescales,
        )

    def __call__(self, time, state):
        reference = self._get_reference_state(time)
        tendencies = get_nudging_tendency(state, reference, self._nudging_timescales)
        ssts = get_reference_surface_temperatures(state, reference)

        reference = {
            f"{key}_reference": reference_state
            for key, reference_state in reference.items()
        }
        return tendencies, ssts, reference


class NudgingStepper(Stepper, LoggingMixin):
    """Stepper for nudging"""

    def __init__(
        self,
        state,
        fv3gfs: Any,
        rank: int,
        config: NudgingConfig,
        timestep: float,
        communicator: fv3gfs.util.CubedSphereCommunicator,
    ):
        self._state = state
        self._timestep: float = timestep
        self.nudger = PureNudger(config, communicator)
        self._tendencies: State = {}
        self._state_updates: State = {}

    def _compute_python_tendency(self) -> Diagnostics:
        (self._tendencies, self._state_updates, diagnostics,) = self.nudger(
            self._state.time, self._state
        )
        return diagnostics

    def _apply_python_to_dycore_state(self) -> Diagnostics:

        diagnostics = compute_nudging_diagnostics(self._state, self._tendencies)
        updated_state: State = apply(self._state, self._tendencies, dt=self._timestep)
        updated_state[TOTAL_PRECIP] = precipitation_sum(
            self._state[TOTAL_PRECIP],
            diagnostics[f"net_moistening_due_to_{self.nudger.name}"],
            self._timestep,
        )
        diagnostics[TOTAL_PRECIP] = updated_state[TOTAL_PRECIP]
        self._state.update(updated_state)
        self._state.update(self._state_updates)
        return diagnostics
