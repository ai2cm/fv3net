"""Top level factory functions

These construct objects like Emulators that require knowledge of static
configuration as well as runtime-only data structures like the model state.
"""
from runtime.monitor import Monitor
from runtime.types import State
from runtime.config import UserConfig
from runtime.emulator.adapter import PrognosticAdapter, get_xarray_emulator


__all__ = ["get_emulator_adapter"]


def get_emulator_adapter(
    config: UserConfig, state: State, timestep: float
) -> PrognosticAdapter:
    monitor = Monitor.from_variables(config.diagnostic_variables, state, timestep)
    return PrognosticAdapter(
        get_xarray_emulator(config.online_emulator), state, monitor
    )
