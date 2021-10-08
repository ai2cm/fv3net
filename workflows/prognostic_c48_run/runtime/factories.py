"""Top level factory functions

These construct objects like Emulators that require knowledge of static
configuration as well as runtime-only data structures like the model state.
"""
from typing import Optional
from runtime.types import State
from runtime.config import UserConfig
from runtime.emulator import PrognosticStepTransformer
from runtime.tendency_prescriber import TendencyPrescriber
from runtime.derived_state import DerivedFV3State
import fv3gfs.util


__all__ = ["get_emulator_adapter", "get_tendency_prescriber"]


def get_emulator_adapter(
    config: UserConfig, state: State, timestep: float
) -> Optional[PrognosticStepTransformer]:
    if config.online_emulator is None:
        return None
    else:
        return PrognosticStepTransformer(
            config.online_emulator,
            state,
            diagnostic_variables=set(config.diagnostic_variables),
            timestep=timestep,
        )


def get_tendency_prescriber(
    config: UserConfig,
    state: DerivedFV3State,
    timestep: float,
    communicator: fv3gfs.util.CubedSphereCommunicator,
) -> Optional[TendencyPrescriber]:
    if config.tendency_prescriber is None:
        return None
    else:
        return TendencyPrescriber(
            config.tendency_prescriber,
            state,
            communicator,
            timestep,
            diagnostic_variables=set(config.diagnostic_variables),
        )
