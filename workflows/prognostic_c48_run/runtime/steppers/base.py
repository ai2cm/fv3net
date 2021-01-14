import logging
from typing import (
    Hashable,
    MutableMapping,
)

import xarray as xr

from runtime.derived_state import DerivedFV3State
from runtime.names import PRECIP_RATE, SPHUM, DELP, AREA
from runtime.diagnostics.machine_learning import compute_baseline_diagnostics

State = MutableMapping[Hashable, xr.DataArray]
Diagnostics = MutableMapping[Hashable, xr.DataArray]

logger = logging.getLogger(__name__)

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


class Stepper(LoggingMixin):
    def __init__(self, rank: int = 0):
        self.rank: int = rank

    @property
    def _state(self):
        return DerivedFV3State(self._fv3gfs)

    def _compute_python_tendency(self) -> Diagnostics:
        return {}

    def _apply_python_to_dycore_state(self) -> Diagnostics:
        return {}

    def _apply_python_to_physics_state(self) -> Diagnostics:
        return {}


class BaselineStepper(Stepper):
    def __init__(self, fv3gfs, rank, states_to_output):
        self._fv3gfs = fv3gfs
        self._states_to_output = states_to_output

    def _compute_python_tendency(self) -> Diagnostics:
        return {}

    def _apply_python_to_dycore_state(self) -> Diagnostics:

        state: State = {name: self._state[name] for name in [PRECIP_RATE, SPHUM, DELP]}
        diagnostics: Diagnostics = compute_baseline_diagnostics(state)
        diagnostics.update({name: self._state[name] for name in self._states_to_output})

        return {
            "area": self._state[AREA],
            "cnvprcp_after_python": self._fv3gfs.get_diagnostic_by_name(
                "cnvprcp"
            ).data_array,
            **diagnostics,
        }

