import dataclasses
from typing import (
    Iterable,
    Optional,
    Set,
    Union,
    Sequence,
    Protocol,
)

import fv3fit.emulation.thermobasis.emulator
from fv3fit.emulation.thermobasis.xarray import get_xarray_emulator
from runtime.masking import get_mask, where_masked
from runtime.monitor import Monitor
from runtime.types import Diagnostics, State, Step

__all__ = ["StepTransformer", "Config"]


def strip_prefix(prefix: str, variables: Iterable[str]) -> Set[str]:
    return {k[len(prefix) :] for k in variables if k.startswith(prefix)}


class Predictor(Protocol):
    """Predictor interface for step transformers."""

    @property
    def input_variables(self) -> Sequence[str]:
        """Variables needed as inputs for prediction."""
        pass

    def predict(self, inputs: State) -> State:
        """Given inputs return state predictions."""
        pass

    def apply(self, prediction: State, state: State):
        """Apply predictions to given state."""
        pass

    def partial_fit(self, inputs: State, state: State):
        """Do partial fit for online training."""
        pass


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

    def predict(self, inputs: State) -> State:
        return self.emulator.predict(inputs)

    def apply(self, prediction: State, state: State):
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
        if self.config.train:
            self.emulator.partial_fit(inputs, state)

    @property
    def input_variables(self) -> Sequence[str]:
        return self.emulator.input_variables


@dataclasses.dataclass
class StepTransformer:
    """Wrap a Step function with an ML prediction

    The wrapped function produces diagnostic outputs prefixed with
    ``self.label + "_"`` and trains/applies the ML model to ``state``
    depending on the user configuration.

    Attributes:
        model: A machine learning model that knows how to apply its updates.
        state: The mutable state being updated.
        label: Used for labeling diagnostic outputs and monitored tendencies.
        diagnostic_variables: The user-requested diagnostic variables.
        timestep: the model timestep in seconds.
    """

    model: Predictor
    state: State
    label: str
    diagnostic_variables: Set[str] = dataclasses.field(default_factory=set)
    timestep: float = 900

    @property
    def monitor(self) -> Monitor:
        return Monitor.from_variables(
            self.diagnostic_variables, self.state, self.timestep
        )

    @property
    def prefix(self) -> str:
        return self.label + "_"

    @property
    def inputs_to_save(self) -> Set[str]:
        return set(strip_prefix(self.prefix, self.diagnostic_variables))

    def transform(self, func: Step) -> Diagnostics:
        inputs: State = {key: self.state[key] for key in self.model.input_variables}
        inputs_to_save: Diagnostics = {
            self.prefix + key: self.state[key] for key in self.inputs_to_save
        }

        before = self.monitor.checkpoint()
        diags = func()

        self.model.partial_fit(inputs, self.state)

        prediction = self.model.predict(inputs)
        for v in before:
            if v not in prediction:
                prediction[v] = self.state[v]

        change_due_to_prediction = self.monitor.compute_change(
            self.label, before, prediction
        )
        self.model.apply(prediction, self.state)
        return {
            **diags,
            **inputs_to_save,
            **change_due_to_prediction,
        }

    def __call__(self, func: Step) -> Step:
        """Transform a function that updates the ``state``
        
        Similar to :py:class:`runtime.monitor.Monitor` but with ML prediction
        capability.

        Args:
            func: a function that updates the State and returns a dictionary of
                diagnostics (usually a method of :py:class:`runtime.loop.TimeLoop`).
        
        Returns:
            A function which calls ``func`` and optionally applies/trains an ML model
            in addition.
        """

        def step() -> Diagnostics:
            return self.transform(func)

        # functools.wraps modifies the type and breaks mypy type checking
        step.__name__ = func.__name__

        return step
