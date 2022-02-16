"""Top level factory functions

These construct objects like Emulators that require knowledge of static
configuration as well as runtime-only data structures like the model state.
"""
from typing import Optional
from runtime.types import State
from runtime.config import UserConfig
from runtime.transformers.core import StepTransformer
from runtime.transformers.tendency_prescriber import TendencyPrescriber
from runtime.derived_state import DerivedFV3State
import runtime.transformers.emulator
import runtime.transformers.fv3fit
import fv3gfs.util


__all__ = ["get_fv3_physics_transformer", "get_tendency_prescriber"]


def get_fv3_physics_transformer(
    config: UserConfig,
    state: State,
    timestep: float,
    communicator: fv3gfs.util.CubedSphereCommunicator,
) -> Optional[StepTransformer]:
    if config.online_emulator is None:
        return None
    elif isinstance(config.online_emulator, runtime.transformers.emulator.Config):
        emulator = runtime.transformers.emulator.Adapter(config.online_emulator)
        return StepTransformer(
            emulator,
            state,
            "emulator",
            diagnostic_variables=set(config.diagnostic_variables),
            timestep=timestep,
        )
    elif isinstance(config.online_emulator, runtime.transformers.fv3fit.Config):
        model = runtime.transformers.fv3fit.Adapter(
            config.online_emulator, timestep, communicator
        )
        return StepTransformer(
            model,
            state,
            "machine_learning",
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
