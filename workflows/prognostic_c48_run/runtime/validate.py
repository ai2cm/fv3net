from datetime import datetime, timedelta
from typing import Any, Mapping, Union

import dacite
import fv3config
from .config import UserConfig, DiagnosticFileConfig
from .diagnostics.fortran import FortranFileConfig


class ConfigValidationError(ValueError):
    pass


def validate_config(config_dict: Mapping[str, Any]):
    """Raises..."""
    validate_chunks(config_dict)


def validate_chunks(config_dict: Mapping[str, Any]):
    user_config = dacite.from_dict(UserConfig, config_dict)
    run_duration = fv3config.get_run_duration(config_dict)
    initial_time = datetime(*config_dict["namelist"]["coupler_nml"]["current_date"])
    timestep = timedelta(seconds=config_dict["namelist"]["coupler_nml"]["dt_atmos"])

    for diag_file_config in user_config.diagnostics:
        _validate_time_chunks(diag_file_config, initial_time, timestep, run_duration)

    for fortran_file_config in user_config.fortran_diagnostics:
        _validate_time_chunks(fortran_file_config, initial_time, timestep, run_duration)


def _validate_time_chunks(
    diag_file_config: Union[DiagnosticFileConfig, FortranFileConfig],
    initial_time: datetime,
    timestep: timedelta,
    run_duration: timedelta,
):
    num_total_timesteps = int(run_duration / timestep)
    all_times = [initial_time + n * timestep for n in range(1, num_total_timesteps + 1)]
    if "time" in diag_file_config.chunks:
        time_container = diag_file_config.times.time_container(initial_time)
        num_output_timesteps = sum(
            time_container.indicator(t) is not None for t in all_times
        )  # type: ignore
        if num_output_timesteps % diag_file_config.chunks["time"] != 0:
            raise ConfigValidationError(
                "Time dimension size must be a multiple of time chunk size. This was "
                f"not the case for an output in '{diag_file_config.name}'."
            )
