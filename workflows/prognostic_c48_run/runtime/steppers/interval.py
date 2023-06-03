import cftime
import dataclasses
from datetime import timedelta
from typing import Tuple, Union
import xarray as xr

from runtime.types import Diagnostics
from runtime.steppers.stepper import Stepper
from runtime.steppers.machine_learning import MachineLearningConfig
from runtime.steppers.prescriber import PrescriberConfig


@dataclasses.dataclass
class IntervalConfig:
    base_config: Union[PrescriberConfig, MachineLearningConfig]
    apply_interval_seconds: int


class IntervalStepper:
    def __init__(self, apply_interval_seconds: float, stepper: Stepper):
        self.start_time = None
        self.interval = timedelta(seconds=apply_interval_seconds)
        self.stepper = stepper

    @property
    def label(self):
        """Label used for naming diagnostics.
        """
        return f"interval_{self.stepper.label}"

    def _need_to_update(self, time: cftime.DatetimeJulian):
        if self.start_time is not None:
            if ((time - self.start_time) % self.interval).seconds != 0:
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
