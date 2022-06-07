import numpy as np
from typing import Callable, Iterable, Optional

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
    def __init__(self, key: str, start: Optional[int], stop: Optional[int]):
        self.key = key
        self.start = start
        self.stop = stop

    def __call__(self, state: FortranState, emulator: FortranState) -> FortranState:
        out = {**emulator}
        use_fortran_state = slice(self.start, self.stop)
        # Fortran state TOA is index 79, and dims are [z, sample]
        out[self.key][use_fortran_state] = state[self.key][use_fortran_state]
        return out
