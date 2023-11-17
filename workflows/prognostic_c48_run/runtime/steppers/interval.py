import cftime
import dataclasses
from datetime import timedelta
from typing import Tuple, Union, Optional, List
import xarray as xr
import logging
import vcm

from runtime.types import Diagnostics
from runtime.steppers.stepper import Stepper
from runtime.steppers.machine_learning import MachineLearningConfig
from runtime.steppers.prescriber import PrescriberConfig
from runtime.nudging import NudgingConfig
from runtime.diagnostics.compute import KG_PER_M2_PER_M
from runtime.names import SPHUM, DELP, TOTAL_PRECIP

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class IntervalConfig:
    """Configuration for interval steppers

    base_config: config for the stepper used at the end of the interval
    apply_interval_seconds: interval in seconds
    offset_seconds: offset from the start of the run in seconds to count
        as start of intervals
    record_fields_before_update: fields listed here will have their values
        immediately before the stepper update recorded in diagnostics
    n_calls: if provided, stop after this many calls to the stepper
        useful for synchronizing reservoir models at the start of runs
    """

    base_config: Union[PrescriberConfig, MachineLearningConfig, NudgingConfig]
    apply_interval_seconds: int
    offset_seconds: int = 0
    record_fields_before_update: Optional[List[str]] = None
    n_calls: Optional[int] = None
    do_total_precip_update: bool = True


class IntervalStepper:
    def __init__(
        self,
        apply_interval_seconds: float,
        stepper: Stepper,
        offset_seconds: float = 0,
        n_calls: Optional[int] = None,
        record_fields_before_update: Optional[List[str]] = None,
        do_total_precip_update: bool = True,
    ):
        self.start_time = None
        self.interval = timedelta(seconds=apply_interval_seconds)
        self.stepper = stepper
        self.offset_seconds = timedelta(seconds=offset_seconds)
        self._record_fields_before_update = record_fields_before_update or []
        self.n_calls = n_calls
        self._call_count = 0
        self._do_total_precip_update = do_total_precip_update

    @property
    def label(self):
        """Label used for naming diagnostics.
        """
        return f"interval_{self.stepper.label}"

    def _need_to_update(self, time: cftime.DatetimeJulian):
        if self.start_time is not None:
            if (
                (time - self.start_time - self.offset_seconds) % self.interval
            ).seconds != 0:
                return False
            else:
                if self.n_calls is None:
                    return True
                else:
                    if self._call_count < self.n_calls:
                        return True
                    else:
                        return False

        else:
            logger.info(f"Setting interval stepper start time to {time}")
            self.start_time = time
            return False

    def get_diagnostics_prior_to_update(self, state):
        return {
            f"{key}_before_interval_update": state[key]
            for key in self._record_fields_before_update
        }

    def __call__(self, time, state):

        if self._need_to_update(time) is False:
            # Diagnostic must be available at all timesteps, not just when
            # the base stepper is called
            diags = self.get_diagnostics_prior_to_update(state)
            _zeros_2d = xr.zeros_like(state[TOTAL_PRECIP])
            _zeros_2d.attrs = {}
            diags.update({f"net_moistening_due_to_{self.label}": _zeros_2d})
            return {}, diags, {}
        else:
            logger.info(f"applying interval stepper at time {time}")
            tendencies, diagnostics, state_updates = self.stepper(time, state)
            diagnostics.update(self.get_diagnostics_prior_to_update(state))
            self._call_count += 1

            if self._do_total_precip_update and SPHUM in state_updates:
                logger.info(f"Updating total precip at time {time}")
                state_updates[TOTAL_PRECIP] = self._get_precipitation_update(
                    state, state_updates
                )
            else:
                logger.info(f"Not updating total precip at time {time} ")
            return tendencies, diagnostics, state_updates

    def _get_precipitation_update(
        self, state, state_updates,
    ):
        corrective_moistening_integral = (
            vcm.mass_integrate(state_updates[SPHUM] - state[SPHUM], state[DELP], "z")
            / KG_PER_M2_PER_M
        )
        total_precip_before_limiter = (
            state[TOTAL_PRECIP] - corrective_moistening_integral
        )
        total_precip = total_precip_before_limiter.where(
            total_precip_before_limiter >= 0, 0
        )
        total_precip.attrs["units"] = "m"
        return total_precip

    def get_diagnostics(self, state, tendency) -> Tuple[Diagnostics, xr.DataArray]:
        diags, moistening = self.stepper.get_diagnostics(state, tendency)
        diags.update(self.get_diagnostics_prior_to_update(state))
        return diags, moistening
