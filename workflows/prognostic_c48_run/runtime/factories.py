"""Top level factory functions
These construct objects like Emulators that require knowledge of static
configuration as well as runtime-only data structures like the model state.
"""
from typing import Optional
from runtime.types import State
from runtime.config import UserConfig
from runtime.emulator import PrognosticAdapter
from runtime.overrider import OverriderAdapter
import fv3gfs.util


__all__ = ["get_emulator_adapter", "get_overrider_adapter"]


def get_emulator_adapter(
    config: UserConfig, state: State, timestep: float
) -> Optional[PrognosticAdapter]:
    if config.online_emulator is None:
        return None
    else:
        return PrognosticAdapter(
            config.online_emulator,
            state,
            diagnostic_variables=set(config.diagnostic_variables),
            timestep=timestep,
        )


def get_overrider_adapter(
    config: UserConfig,
    state: State,
    communicator: fv3gfs.util.CubedSphereCommunicator,
    timestep: float,
) -> Optional[OverriderAdapter]:
    if config.overrider is None:
        return None
    else:
        return OverriderAdapter(
            config.overrider,
            state,
            communicator,
            timestep,
            diagnostic_variables=set(config.diagnostic_variables),
        )
