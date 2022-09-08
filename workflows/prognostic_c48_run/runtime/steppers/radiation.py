import dataclasses
from typing import Optional, Literal, Union, Tuple, Hashable
import cftime
import xarray as xr
from runtime.steppers.machine_learning import MachineLearningConfig, PureMLStepper
from runtime.steppers.prescriber import PrescriberConfig, Prescriber
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
        self,
        radiation: Radiation,
        input_generator: Optional[Union[PureMLStepper, Prescriber]] = None,
    ):
        self._radiation = radiation
        self._input_generator = input_generator

    def __call__(
        self, time: cftime.DatetimeJulian, state: State,
    ):
        if self._input_generator is not None:
            state = self._generate_inputs(state, time)
        diagnostics = self._radiation(time, state)
        return {}, diagnostics, {}

    def get_diagnostics(self, state, tendency) -> Tuple[Diagnostics, xr.DataArray]:
        return {}, xr.DataArray()

    def get_momentum_diagnostics(self, state, tendency) -> Diagnostics:
        return {}

    def _generate_inputs(self, state: State, time: cftime.DatetimeJulian) -> State:
        if self._input_generator is not None:
            _, _, state_updates = self._input_generator(time, state)
            return MergedState(state, state_updates)
        else:
            return state


class MergedState(State):
    def __init__(self, state: State, overriding_state: State):
        self._state = state
        self._overriding_state = overriding_state

    def __getitem__(self, key: Hashable) -> xr.DataArray:
        if key in self._overriding_state:
            return self._overriding_state[key]
        elif key in self._state:
            return self._state[key]
        else:
            raise KeyError("Key is in neither state mapping.")

    def keys(self):
        return set(self._state.keys()) | set(self._overriding_state.keys())

    def __delitem__(self, key: Hashable):
        raise NotImplementedError()

    def __setitem__(self, key: Hashable, value: xr.DataArray):
        raise NotImplementedError()

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(self.keys())
