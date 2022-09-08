import dataclasses
from typing import (
    Optional,
    Literal,
    Union,
    Tuple,
)
import cftime
import xarray as xr
from runtime.steppers.machine_learning import MachineLearningConfig
from runtime.steppers.prescriber import PrescriberConfig
from runtime.types import State, Diagnostics
from radiation import Radiation


@dataclasses.dataclass
class RadiationConfig:
    """"""

    kind: Literal["python"]
    input_generator: Optional[Union[PrescriberConfig, MachineLearningConfig]] = None


class RadiationStepper:

    label = "radiation"

    def __init__(
        self, radiation: Radiation,
    ):
        self._radiation = radiation

    def __call__(
        self, time: cftime.DatetimeJulian, state: State,
    ):
        diagnostics = self._radiation(time, state)
        return {}, diagnostics, {}

    def get_diagnostics(self, state, tendency) -> Tuple[Diagnostics, xr.DataArray]:
        return {}, xr.DataArray()

    def get_momentum_diagnostics(self, state, tendency) -> Diagnostics:
        return {}
