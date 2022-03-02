"""Top level factory functions

These construct objects like Emulators that require knowledge of static
configuration as well as runtime-only data structures like the model state.
"""
import logging
from typing import Optional, Callable, Sequence, Mapping
from datetime import timedelta
import xarray as xr
import cftime
import vcm
from vcm.limit import DatasetQuantileLimiter
from loaders.mappers import open_zarr
from runtime.types import State
from runtime.config import UserConfig
from runtime.transformers.core import StepTransformer
from runtime.transformers.tendency_prescriber import TendencyPrescriber
from runtime.steppers.prescriber import PrescriberConfig, Prescriber
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
        tendency_variables = list(prescriber_config.variables.values())
        if communicator.tile.rank == 0:
            if communicator.rank == 0:
                logger.debug(
                    f"Opening tendency override from: {prescriber_config.mapper_config}"
                )
            mapper = prescriber_config.mapper_config.load_mapper()
        else:
            mapper = {}

        if isinstance(prescriber_config.limit_quantiles, dict):
            if prescriber_config.reference_initial_time is None:
                raise ValueError(
                    "TendencyPrescriber reference_initial_time must be specified if "
                    "limit_quantiles are specified."
                )
            limiter: Optional[DatasetQuantileLimiter] = _get_fitted_limiter(
                mapper,
                prescriber_config.reference_initial_time,
                prescriber_config.limit_quantiles,
                limit_only=tendency_variables,
            )
        else:
            limiter = None

        time_lookup_function = _get_time_lookup_function(
            mapper,
            tendency_variables,
            prescriber_config.reference_initial_time,
            prescriber_config.reference_frequency_seconds,
            limiter=limiter,
        )

        return TendencyPrescriber(
            state,
            communicator,
            timestep,
            prescriber_config.variables,
            time_lookup_function,
            diagnostic_variables=set(config.diagnostic_variables),
        )


def _get_time_lookup_function(
    mapper: Mapping[str, xr.Dataset],
    variables: Sequence[str],
    initial_time: Optional[str] = None,
    frequency_seconds: float = 900.0,
    limiter: Optional[DatasetQuantileLimiter] = None,
) -> Callable[[cftime.DatetimeJulian], State]:
    def time_lookup_function(time: cftime.DatetimeJulian) -> State:
        timestamp = vcm.encode_time(time)
        ds = mapper[timestamp]
        if limiter is not None:
            ds = limiter.transform(ds)
        return {var: ds[var].load() for var in variables}

    if initial_time is not None:
        initial_time = label_to_time(initial_time)
        return time_interpolate_func(
            time_lookup_function, timedelta(seconds=frequency_seconds), initial_time
        )
    else:
        return time_lookup_function


def _get_fitted_limiter(
    mapper: Mapping[str, xr.Dataset],
    initial_time: str,
    limit_quantiles: Mapping[str, float],
    limit_only: Optional[Sequence[str]] = None,
) -> DatasetQuantileLimiter:

    sample_tendencies = mapper[initial_time]

    limiter = DatasetQuantileLimiter(
        limit_quantiles["upper"], limit_quantiles["lower"], limit_only=limit_only,
    )

    logger.debug(f"Fitting dataset limiter with limits={limit_quantiles}")
    return limiter.fit(sample_tendencies, feature_dims=["z", "tile"])


def get_prescriber(
    config: PrescriberConfig, communicator: pace.util.CubedSphereCommunicator
) -> Prescriber:
    if communicator.tile.rank == 0:
        if communicator.rank == 0:
            logger.info(f"Setting up dataset for state setting: {config.dataset_key}")
        mapper = open_zarr(config.dataset_key, config.consolidated)
    else:
        mapper = {}
    time_lookup_function = _get_time_lookup_function(
        mapper,
        config.variables,
        config.reference_initial_time,
        config.reference_frequency_seconds,
    )
    return Prescriber(communicator, time_lookup_function)
