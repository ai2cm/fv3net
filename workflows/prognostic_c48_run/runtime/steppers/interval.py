import cftime
from datetime import timedelta
from typing import Union, Tuple
import xarray as xr

from runtime.steppers.machine_learning import PureMLStepper
from runtime.steppers.prescriber import Prescriber
from runtime.types import Diagnostics


class IntervalStepper:
    def __init__(
        self, apply_interval_seconds: float, stepper: Union[Prescriber, PureMLStepper]
    ):
        self.start_time = None
        self.interval = timedelta(seconds=apply_interval_seconds)
        self.stepper = stepper

    def _need_to_update(self, time: cftime.DatetimeJulian):
        if self.start_time:
            if (
                (time - self.start_time) % self.interval
            ).seconds != 0 or time == self.start_time:
                return False
            else:
                return True

        else:
            self.start_time = time
            return False

    def __call__(self, time, state):
        if self._need_to_update(time) is False:
            return {}, {}, {}
        else:
            return self.stepper(time, state)

    def get_diagnostics(self, state, tendency) -> Tuple[Diagnostics, xr.DataArray]:
        return self.stepper.get_diagnostics(state, tendency)
