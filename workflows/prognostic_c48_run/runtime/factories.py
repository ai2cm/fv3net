"""Top level factory functions

These construct objects like Emulators that require knowledge of static
configuration as well as runtime-only data structures like the model state.
"""
import logging
from typing import Optional, Callable, Sequence
from datetime import timedelta
import cftime
import loaders
import vcm
from runtime.types import State
from runtime.config import UserConfig
from runtime.transformers.core import StepTransformer
from runtime.transformers.tendency_prescriber import TendencyPrescriber
from runtime.interpolate import time_interpolate_func, label_to_time
from runtime.derived_state import DerivedFV3State
import runtime.transformers.emulator
import runtime.transformers.fv3fit
import pace.util

logger = logging.getLogger(__name__)


__all__ = ["get_fv3_physics_transformer", "get_tendency_prescriber"]


def get_fv3_physics_transformer(
    config: UserConfig, state: State, timestep: float
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
        model = runtime.transformers.fv3fit.Adapter(config.online_emulator, timestep)
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
    communicator: pace.util.CubedSphereCommunicator,
) -> Optional[TendencyPrescriber]:
    if config.tendency_prescriber is None:
        return None
    else:
        prescriber_config = config.tendency_prescriber
        mapper_config = prescriber_config.mapper_config
        tendency_variables = list(prescriber_config.variables.values())
        if communicator.rank == 0:
            logger.debug(f"Opening tendency override from: {mapper_config}")
        mapper_function = _get_mapper_function(
            mapper_config,
            tendency_variables,
            prescriber_config.reference_initial_time,
            prescriber_config.reference_frequency_seconds,
        )
        return TendencyPrescriber(
            state,
            communicator,
            timestep,
            prescriber_config.variables,
            mapper_function,
            limit_quantiles=prescriber_config.limit_quantiles,
            diagnostic_variables=set(config.diagnostic_variables),
        )


def _get_mapper_function(
    mapper_config: loaders.MapperConfig,
    tendency_variables: Sequence[str],
    initial_time: Optional[str] = None,
    frequency_seconds: float = 900.0,
) -> Callable[[cftime.DatetimeJulian], State]:

    mapper = mapper_config.load_mapper()

    def mapper_function(time: cftime.DatetimeJulian) -> State:
        timestamp = vcm.encode_time(time)
        ds = mapper[timestamp]
        return {var: ds[var] for var in tendency_variables}

    if initial_time is not None:
        initial_time = label_to_time(initial_time)
        return_func = time_interpolate_func(
            mapper_function, timedelta(seconds=frequency_seconds), initial_time
        )
    else:
        return_func = mapper_function

    return return_func
