import logging

import xarray as xr
from runtime.diagnostics.machine_learning import compute_baseline_diagnostics
from runtime.names import TENDENCY_TO_STATE_NAME
from runtime.types import State

logger = logging.getLogger(__name__)

KG_PER_M2_PER_M = 1000.0


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


class BaselineStepper:
    net_moistening = "doesn't matter"

    def __call__(self, time, state):
        return {}, {}, {}, None, {}

    def get_diagnostics(self, state, tendency):
        return compute_baseline_diagnostics(state)

    def get_momentum_diagnostics(self, state, tendency):
        return {}

    def apply(self, state, tendency, dt):
        return apply(state, tendency, dt)


def apply(state: State, tendency: State, dt: float) -> State:
    """Given state and tendency prediction, return updated state.
    Returned state only includes variables updated by ML model."""

    with xr.set_options(keep_attrs=True):
        updated = {}
        for name in tendency:
            state_name = TENDENCY_TO_STATE_NAME.get(name, name)
            updated[state_name] = state[state_name] + tendency[name] * dt
    return updated  # type: ignore


def precipitation_sum(
    physics_precip: xr.DataArray, column_dq2: xr.DataArray, dt: float
) -> xr.DataArray:
    """Return sum of physics precipitation and ML-induced precipitation. Output is
    thresholded to enforce positive precipitation.

    Args:
        physics_precip: precipitation from physics parameterizations [m]
        column_dq2: column-integrated moistening from ML [kg/m^2/s]
        dt: physics timestep [s]

    Returns:
        total precipitation [m]"""
    m_per_mm = 1 / 1000
    ml_precip = -column_dq2 * dt * m_per_mm  # type: ignore
    total_precip = physics_precip + ml_precip
    total_precip = total_precip.where(total_precip >= 0, 0)
    total_precip.attrs["units"] = "m"
    return total_precip


def precipitation_rate(
    precipitation_accumulation: xr.DataArray, dt: float
) -> xr.DataArray:
    """Return precipitation rate from a precipitation accumulation and timestep
    
    Args:
        precipitation_accumulation: precipitation accumulation [m]
        dt: timestep over which accumulation occurred [s]

    Returns:
        precipitation rate [kg/m^s/s]"""
    precipitation_rate: xr.DataArray = (
        KG_PER_M2_PER_M * precipitation_accumulation / dt  # type: ignore
    )
    precipitation_rate.attrs["units"] = "kg/m^2/s"
    return precipitation_rate
