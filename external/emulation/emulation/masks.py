import numpy as np
from typing import Callable, Iterable, Optional, Union

from emulation._typing import FortranState

Mask = Callable[[FortranState, FortranState], FortranState]


def compose_masks(funcs: Iterable[Mask]) -> Mask:
    """Compose multiple masks first masks are applied first"""

    func_list = list(funcs)

    def composed(state: FortranState, emulator: FortranState):
        out: FortranState = emulator
        for func in func_list:
            out = func(state, out)
        return out

    return composed


class RangeMask:
    def __init__(
        self, key: str, min: Optional[float] = None, max: Optional[float] = None
    ) -> None:
        self.min = min
        self.max = max
        self.key = key

    def __call__(self, state: FortranState, emulator: FortranState) -> FortranState:
        out = {**emulator}
        if self.min is not None:
            out[self.key] = np.maximum(out[self.key], self.min)

        if self.max is not None:
            out[self.key] = np.minimum(out[self.key], self.max)

        return out


class LevelMask:
    def __init__(
        self,
        key: str,
        start: Optional[int],
        stop: Optional[int],
        fill_value: Union[float, str, None] = None,
    ):
        self.key = key
        self.start = start
        self.stop = stop
        self.fill_value = fill_value

    def __call__(self, state: FortranState, emulator: FortranState) -> FortranState:
        use_fortran_state = slice(self.start, self.stop)
        # Fortran state TOA is index 79, and dims are [z, sample]
        emulator_field = np.copy(emulator[self.key])

        # Currently, fortran fields pushed into python state are 64bit floats
        # while the emulator output is float32, since there are no post-hoc adjustments
        # for precpd, this lead to noise in the tendencies estimated from the
        # masked levels due to 32 -> 64 casting, this hack resolves
        if emulator_field.dtype != np.float64:
            emulator_field = emulator_field.astype(np.float64)

        if self.fill_value is None:
            emulator_field[use_fortran_state] = state[self.key][use_fortran_state]
        elif isinstance(self.fill_value, str):
            emulator_field[use_fortran_state] = state[self.fill_value][
                use_fortran_state
            ]
        elif isinstance(self.fill_value, float):
            emulator_field[use_fortran_state] = self.fill_value

        return {**emulator, self.key: emulator_field}
