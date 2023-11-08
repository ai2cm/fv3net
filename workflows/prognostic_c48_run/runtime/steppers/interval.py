import cftime
import dataclasses
from datetime import timedelta
from typing import Tuple, Union, Optional, List
import xarray as xr
import logging

from runtime.types import Diagnostics
from runtime.steppers.stepper import Stepper
from runtime.steppers.machine_learning import MachineLearningConfig
from runtime.steppers.prescriber import PrescriberConfig
from runtime.nudging import NudgingConfig


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


class IntervalStepper:
    def __init__(
        self,
        apply_interval_seconds: float,
        stepper: Stepper,
        offset_seconds: float = 0,
        n_calls: Optional[int] = None,
        record_fields_before_update: Optional[List[str]] = None,
    ):
        self.start_time = None
        self.interval = timedelta(seconds=apply_interval_seconds)
        self.stepper = stepper
        self.offset_seconds = timedelta(seconds=offset_seconds)
        self._record_fields_before_update = record_fields_before_update or []
        self.n_calls = n_calls
        self._call_count = 0

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
            return {}, self.get_diagnostics_prior_to_update(state), {}
        else:
            logger.info(f"applying interval stepper at time {time}")
            tendencies, diagnostics, state_updates = self.stepper(time, state)
            diagnostics.update(self.get_diagnostics_prior_to_update(state))
            self._call_count += 1
            return tendencies, diagnostics, state_updates

    def get_diagnostics(self, state, tendency) -> Tuple[Diagnostics, xr.DataArray]:
        diags, moistening = self.stepper.get_diagnostics(state, tendency)
        diags.update(self.get_diagnostics_prior_to_update(state))
        return diags, moistening
