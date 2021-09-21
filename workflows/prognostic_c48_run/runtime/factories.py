"""Top level factory functions

These construct objects like Emulators that require knowledge of static
configuration as well as runtime-only data structures like the model state.
"""
from typing import Optional
from runtime.types import State
from runtime.config import UserConfig
from runtime.emulator import PrognosticAdapter
from runtime.ml_prescriber import MLPrescriberAdapter


__all__ = ["get_emulator_adapter"]


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


def get_ml_prescriber_adapter(
    config: UserConfig, state: State, timestep: float
) -> Optional[MLPrescriberAdapter]:
    if config.ml_prescriber is None:
        return None
    else:
        return MLPrescriberAdapter(
            config.ml_prescriber,
            state,
            timestep,
            diagnostic_variables=set(config.diagnostic_variables),
        )
