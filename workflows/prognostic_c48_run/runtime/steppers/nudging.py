import functools
from typing import Any, List, Sequence, Optional

import fv3gfs.util

from runtime.steppers.base import (
    Stepper,
    State,
    Diagnostics,
    apply,
    precipitation_sum,
    precipitation_rate,
    LoggingMixin,
)

from runtime.diagnostics.machine_learning import compute_nudging_diagnostics

from runtime.nudging import (
    nudging_timescales_from_dict,
    setup_get_reference_state,
    get_nudging_tendency,
    set_state_sst_to_reference,
    NudgingConfig,
)

from runtime.names import (
    DELP,
    TOTAL_PRECIP,
    PRECIP_RATE,
    AREA,
)


SST_NAME = "ocean_surface_temperature"
TSFC_NAME = "surface_temperature"
MASK_NAME = "land_sea_mask"


class NudgingStepper(Stepper, LoggingMixin):
    """Stepper for nudging
    """

    def __init__(
        self,
        fv3gfs: Any,
        rank: int,
        config: NudgingConfig,
        timestep: float,
        states_to_output: Sequence[str],
        communicator: fv3gfs.util.CubedSphereCommunicator,
    ):

        self._states_to_output = states_to_output

        self._fv3gfs = fv3gfs
        self.rank: int = rank
        self._timestep: float = timestep

        self._nudging_timescales = nudging_timescales_from_dict(config.timescale_hours)
        self._get_reference_state = setup_get_reference_state(
            config,
            self.nudging_variables + [SST_NAME, TSFC_NAME],
            self._fv3gfs.get_tracer_metadata(),
            communicator,
        )
        self._get_nudging_tendency = functools.partial(
            get_nudging_tendency, nudging_timescales=self._nudging_timescales,
        )
        self._tendencies_to_apply_to_dycore_state: State = {}

    @property
    def nudging_variables(self) -> List[str]:
        return list(self._nudging_timescales)

    def _compute_python_tendency(
        self, diagnostics: Optional[Diagnostics]
    ) -> Diagnostics:

        self._log_debug("Computing nudging tendencies")
        variables: List[str] = self.nudging_variables + [
            SST_NAME,
            TSFC_NAME,
            MASK_NAME,
        ]
        state: State = {name: self._state[name] for name in variables}
        reference = self._get_reference_state(self._state.time)
        set_state_sst_to_reference(state, reference)
        self._tendencies_to_apply_to_dycore_state = self._get_nudging_tendency(
            state, reference
        )

        return {
            f"{key}_reference": reference_state
            for key, reference_state in reference.items()
        }

    def _apply_python_to_dycore_state(self) -> Diagnostics:

        diagnostics: Diagnostics = {}

        variables: List[str] = self.nudging_variables + [
            TOTAL_PRECIP,
            PRECIP_RATE,
            DELP,
        ]
        self._log_debug(f"Getting state variables: {variables}")
        state: State = {name: self._state[name] for name in variables}
        tendency: State = self._tendencies_to_apply_to_dycore_state

        diagnostics.update(compute_nudging_diagnostics(state, tendency))
        updated_state: State = apply(state, tendency, dt=self._timestep)
        updated_state[TOTAL_PRECIP] = precipitation_sum(
            state[TOTAL_PRECIP],
            diagnostics["net_moistening_due_to_nudging"],
            self._timestep,
        )

        self._log_debug("Setting Fortran State")
        self._state.update(updated_state)

        diagnostics.update({name: self._state[name] for name in self._states_to_output})

        return {
            "area": self._state[AREA],
            "cnvprcp_after_python": self._fv3gfs.get_diagnostic_by_name(
                "cnvprcp"
            ).data_array,
            "total_precipitation_rate": precipitation_rate(
                updated_state[TOTAL_PRECIP], self._timestep
            ),
            **diagnostics,
        }
