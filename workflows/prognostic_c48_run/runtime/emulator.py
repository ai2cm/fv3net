import dataclasses
from typing import Hashable, Iterable, MutableMapping, Optional, Set, Union

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

    config: Config
    state: State
    emulator_prefix: str = "emulator_"
    diagnostic_variables: Set[str] = dataclasses.field(default_factory=set)
    timestep: float = 900

    def __post_init__(self: "PrognosticStepTransformer"):
        self.emulator = get_xarray_emulator(self.config.emulator)

    @property
    def monitor(self) -> Monitor:
        return Monitor.from_variables(
            self.diagnostic_variables, self.state, self.timestep
        )

    @property
    def inputs_to_save(self) -> Set[str]:
        return set(strip_prefix(self.emulator_prefix, self.diagnostic_variables))

    def emulate(self, name: str, func: Step) -> Diagnostics:
        inputs = {key: self.state[key] for key in self.emulator.input_variables}

        inputs_to_save: Diagnostics = {
            self.emulator_prefix + key: self.state[key] for key in self.inputs_to_save
        }
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
            updated_state = where_masked(
                self.state,
                emulator_prediction,
                compute_mask=get_mask(
                    self.config.mask_kind, self.config.ignore_humidity_below
                ),
            )
            self.state.update(updated_state)

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
