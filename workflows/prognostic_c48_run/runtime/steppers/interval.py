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
from runtime.diagnostics.compute import precipitation_sum, precipitation_rate
from runtime.names import SPHUM, DELP, TOTAL_PRECIP, TOTAL_PRECIP_RATE

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
            diags.update(
                {
                    f"net_moistening_due_to_{self.label}": xr.zeros_like(
                        state[TOTAL_PRECIP]
                    )
                }
            )
            return {}, diags, {}
        else:
            logger.info(f"applying interval stepper at time {time}")
            tendencies, diagnostics, state_updates = self.stepper(time, state)
            diagnostics.update(self.get_diagnostics_prior_to_update(state))
            self._call_count += 1

            if self._do_total_precip_update and SPHUM in state_updates:
                logger.info(f"updating total precip at time {time}")
                net_moistening = self._net_moistening(
                    sphum_before_update=state[SPHUM],
                    sphum_after_update=state_updates[SPHUM],
                    delp=state[DELP],
                )
                diagnostics[f"net_moistening_due_to_{self.label}"] = net_moistening
                total_precipitation_update = precipitation_sum(
                    state[TOTAL_PRECIP], net_moistening, self.interval.total_seconds()
                )
                diagnostics[TOTAL_PRECIP_RATE] = precipitation_rate(
                    total_precipitation_update, self.interval.total_seconds()
                )
                state_updates[TOTAL_PRECIP] = total_precipitation_update
            else:
                logger.info(f"Not updating total precip  ")
            return tendencies, diagnostics, state_updates

    def _net_moistening(self, sphum_before_update, sphum_after_update, delp):
        _dQ2 = (
            sphum_after_update - sphum_before_update
        ) / self.interval.total_seconds()
        return vcm.mass_integrate(_dQ2, delp, "z")

    def get_diagnostics(self, state, tendency) -> Tuple[Diagnostics, xr.DataArray]:
        diags, moistening = self.stepper.get_diagnostics(state, tendency)
        diags.update(self.get_diagnostics_prior_to_update(state))
        return diags, moistening
