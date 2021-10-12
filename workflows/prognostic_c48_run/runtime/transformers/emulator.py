import dataclasses
from typing import Optional, Union, Sequence

import fv3fit.emulation.thermobasis.emulator
from fv3fit.emulation.thermobasis.xarray import get_xarray_emulator
from runtime.masking import get_mask, where_masked
from runtime.types import State

__all__ = ["Config", "Adapter"]


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
class Adapter:
    config: Config

    def __post_init__(self: "Adapter"):
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
