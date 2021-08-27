import dataclasses
import xarray as xr
from typing import (
    Callable,
    Hashable,
    Mapping,
    MutableMapping,
    Optional,
    Union,
)

from fv3fit.emulation.thermobasis.emulator import Config as EmulatorConfig
from fv3fit.emulation.thermobasis.xarray import get_xarray_emulator
from runtime.monitor import Monitor
from runtime.names import SPHUM, DELP
from runtime.types import State, Diagnostics, Step


@dataclasses.dataclass
class Config:
    """
    Attributes:
        checkpoint: Either path to model artifact in Weights and biases
            "<entity>/<project>/<name>:tag" to be loaded, or the configurations
            for a new emulator object.
    """

    emulator: Union[str, EmulatorConfig] = dataclasses.field(
        default_factory=EmulatorConfig
    )
    # online parameters
    online: bool = False
    train: bool = True
    # will ignore the emulator for any z larger than this value
    # remember higher z is lower in the atmosphere hence "below"
    ignore_humidity_below: Optional[int] = None


@dataclasses.dataclass
class PrognosticAdapter:
    """Wrap a Step function with an emulator

    The wrapped function produces diagnostic outputs prefixed with
    ``self.emulator_prefix_`` and trains/applies the emulator to ``state``
    depending on the user configuration.
    """

    config: Config
    state: State
    monitor: Monitor
    emulator_prefix: str = "emulator_"

    def __post_init__(self: "PrognosticAdapter"):
        self.emulator = get_xarray_emulator(self.config.emulator)

    def emulate(self, name: str, func: Step) -> Diagnostics:
        inputs = {key: self.state[key] for key in self.emulator.input_variables}

        inputs_to_save = {self.emulator_prefix + key: self.state[key] for key in inputs}
        before = self.monitor.checkpoint()
        diags = func()
        change_in_func = self.monitor.compute_change(name, before, self.state)

        if self.config.train:
            self.emulator.partial_fit(inputs, self.state)
        emulator_prediction = self.emulator.predict(inputs)

        emulator_after: MutableMapping[Hashable, xr.DataArray] = {
            DELP: before[DELP],
            **emulator_prediction,
        }

        # insert state variables not predicted by the emulator
        for v in before:
            if v not in emulator_after:
                emulator_after[v] = self.state[v]

        changes = self.monitor.compute_change("emulator", before, emulator_after)

        if self.config.online:
            update_state_with_emulator(
                self.state,
                emulator_prediction,
                ignore_humidity_below=self.config.ignore_humidity_below,
            )
        return {**diags, **inputs_to_save, **changes, **change_in_func}

    def __call__(self, name: str, func: Step) -> Step:
        def step() -> Diagnostics:
            return self.emulate(name, func)

        # functools.wraps modifies the type and breaks mypy type checking
        step.__name__ = func.__name__

        return step


def _update_state_with_emulator(
    state: MutableMapping[Hashable, xr.DataArray],
    src: Mapping[Hashable, xr.DataArray],
    from_orig: Callable[[Hashable, xr.DataArray], xr.DataArray],
) -> None:
    """
    Args:
        state: the mutable state object
        src: updates to put into state
        from_orig: a function returning a mask. Where this mask is True, the
            original state array will be used.

    """
    for key in src:
        arr = state[key]
        mask = from_orig(key, arr)
        state[key] = arr.where(mask, src[key].variable)


@dataclasses.dataclass
class from_orig:
    ignore_humidity_below: Optional[int] = None

    def __call__(self, name: Hashable, arr: xr.DataArray) -> xr.DataArray:
        if name == SPHUM:
            if self.ignore_humidity_below is not None:
                return arr.z > self.ignore_humidity_below
            else:
                return xr.DataArray(False)
        else:
            return xr.DataArray(True)


def update_state_with_emulator(
    state: MutableMapping[Hashable, xr.DataArray],
    src: Mapping[Hashable, xr.DataArray],
    ignore_humidity_below: Optional[int] = None,
) -> None:
    return _update_state_with_emulator(state, src, from_orig(ignore_humidity_below))
