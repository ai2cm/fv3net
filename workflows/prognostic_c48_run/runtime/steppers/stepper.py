import xarray as xr
from typing import Protocol, Tuple
from runtime.types import Diagnostics, State, Tendencies


class Stepper(Protocol):
    """Stepper interface

    Steppers know the difference between tendencies, diagnostics, and
    in-place state updates, but they do not know how and when these updates
    will be applied.

    Note:
        Uses typing_extensions.Protocol to avoid the need for explicit sub-typing

    """

    @property
    def label(self) -> str:
        """Label used for naming diagnostics.
        """
        pass

    def __call__(self, time, state) -> Tuple[Tendencies, Diagnostics, State]:
        return {}, {}, {}

    def get_diagnostics(self, state, tendency) -> Tuple[Diagnostics, xr.DataArray]:
        """Return diagnostics mapping and net moistening array."""
        return {}, xr.DataArray()
