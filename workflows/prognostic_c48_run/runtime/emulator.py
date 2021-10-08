import dataclasses
from typing import Hashable, Iterable, MutableMapping, Optional, Set, Union, Sequence

import fv3fit.emulation.thermobasis.emulator
import xarray as xr
from fv3fit.emulation.thermobasis.xarray import get_xarray_emulator
from runtime.masking import get_mask, where_masked
from runtime.monitor import Monitor
from runtime.names import DELP
from runtime.types import Diagnostics, State, Step

__all__ = ["PrognosticStepTransformer", "Config"]


def strip_prefix(prefix: str, variables: Iterable[str]) -> Set[str]:
    return {k[len(prefix) :] for k in variables if k.startswith(prefix)}


@dataclasses.dataclass
class Config:
    """
    Attributes:
        emulator: Either a path to a model to-be-loaded or an emulator
            configuration specifying model parameters.
        online: if True, the emulator will be replace fv3 physics for all
            humidities above level ``ignore_humidity_below``.
        train: if True, each timestep will be used to train the model.
        mask_kind: the type of mask. Defers to the functions
            ``runtime.masking.compute_{mask_kind}``. The default does not mask any
            of the emulator predictions.
        ignore_humidity_below: if mask_kind ="default", then use the fv3 physics
            instead of the emulator above this level.
    """

    emulator: Union[
        str, fv3fit.emulation.thermobasis.emulator.Config
    ] = dataclasses.field(default_factory=fv3fit.emulation.thermobasis.emulator.Config)
    # online parameters
    online: bool = False
    train: bool = True
    # will ignore the emulator for any z larger than this value
    # remember higher z is lower in the atmosphere hence "below"
    ignore_humidity_below: Optional[int] = None
    mask_kind: str = "default"


@dataclasses.dataclass
class EmulatorAdapter:
    config: Config

    def __post_init__(self: "EmulatorAdapter"):
        self.emulator = get_xarray_emulator(self.config.emulator)
        self.online = self.config.online

    def predict(self, inputs: State) -> State:
        return self.emulator.predict(inputs)

    def apply(self, state: State, prediction: State):
        if self.config.online:
            updated_state = where_masked(
                state,
                prediction,
                compute_mask=get_mask(
                    self.config.mask_kind, self.config.ignore_humidity_below
                ),
            )
            state.update(updated_state)

    def partial_fit(self, inputs: State, state: State):
        self.emulator.partial_fit(inputs, state)

    @property
    def input_variables(self) -> Sequence[str]:
        return self.emulator.input_variables


@dataclasses.dataclass
class PrognosticStepTransformer:
    """Wrap a Step function with an ML prediction

    The wrapped function produces diagnostic outputs prefixed with
    ``self.emulator_prefix_`` and trains/applies the emulator to ``state``
    depending on the user configuration.

    Attributes:
        state: The mutable state being updated.
        monitor: A Monitor object to use for saving outputs.
        emulator_prefix: the prefix to use adjust the outputted variable names.
        diagnostic_variables: the user-requested diagnostic variables, will be
            searched for inputs starting with ``emulator_prefix``.
        timestep: the model timestep in seconds
        inputs_to_save: the set of diagnostics that will be produced by the
            emulated step function.
    
    """

    model: EmulatorAdapter
    state: State
    label: str = "emulator"
    diagnostic_variables: Set[str] = dataclasses.field(default_factory=set)
    timestep: float = 900

    @property
    def monitor(self) -> Monitor:
        return Monitor.from_variables(
            self.diagnostic_variables, self.state, self.timestep
        )

    @property
    def inputs_to_save(self) -> Set[str]:
        return set(strip_prefix(self.label + "_", self.diagnostic_variables))

    def emulate(self, name: str, func: Step) -> Diagnostics:
        inputs: State = {key: self.state[key] for key in self.model.input_variables}

        inputs_to_save: Diagnostics = {
            self.label + "_" + key: self.state[key] for key in self.inputs_to_save
        }
        before = self.monitor.checkpoint()
        diags = func()
        change_in_func = self.monitor.compute_change(name, before, self.state)

        if hasattr(self.model, "partial_fit") and self.model.config.train:
            self.model.partial_fit(inputs, self.state)
        prediction = self.model.predict(inputs)

        state_after: MutableMapping[Hashable, xr.DataArray] = {
            DELP: before[DELP],
            **prediction,
        }

        # insert state variables not predicted by the model
        for v in before:
            if v not in state_after:
                state_after[v] = self.state[v]

        changes = self.monitor.compute_change(self.label, before, state_after)

        self.model.apply(self.state, prediction)

        return {**diags, **inputs_to_save, **changes, **change_in_func}

    def __call__(self, name: str, func: Step) -> Step:
        """Emulate a function that updates the ``state``
        
        Similar to :py:class:`runtime.monitor.Monitor` but with prognostic emulation
        capability.

        Args:
            name: The name of the emulator
            func: a function that updates the State and returns a dictionary of
                diagnostics (usually a method of :py:class:`runtime.loop.TimeLoop`).
        
        Returns:
            emulated_func: a function which observes the change to
                ``self.state`` done by ``func`` and optionally applies/trains an ML
                emulator.

        """

        def step() -> Diagnostics:
            return self.emulate(name, func)

        # functools.wraps modifies the type and breaks mypy type checking
        step.__name__ = func.__name__

        return step
