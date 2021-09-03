"""Top level factory functions

These construct objects like Emulators that require knowledge of static
configuration as well as runtime-only data structures like the model state.
"""
from runtime.types import State
from runtime.config import UserConfig
from runtime.emulator import PrognosticAdapter


__all__ = ["get_emulator_adapter"]


def get_emulator_adapter(
    config: UserConfig, state: State, timestep: float
) -> PrognosticAdapter:
    return PrognosticAdapter(
        config.online_emulator,
        state,
        diagnostic_variables=set(config.diagnostic_variables),
        timestep=timestep,
    )
