from datetime import datetime, timedelta
import os
from typing import Mapping

import dacite
import fv3config
import fsspec
from runtime.config import UserConfig


class ConfigValidationError(ValueError):
    pass


def validate_chunks(config_dict: Mapping):
    user_config = dacite.from_dict(UserConfig, config_dict)
    run_duration = fv3config.get_run_duration(config_dict)
    initial_time = datetime(*config_dict["namelist"]["coupler_nml"]["current_date"])
    timestep = timedelta(seconds=config_dict["namelist"]["coupler_nml"]["dt_atmos"])
    physics_output_interval = timedelta(
        hours=config_dict["namelist"]["atmos_model_nml"]["fhout"]
    )

    if isinstance(config_dict["diag_table"], str):
        # assume diag_table item is path to a file
        with fsspec.open(config_dict["diag_table"], "r") as f:
            diag_table = fv3config.DiagTable.from_str(f.read())
    else:
        diag_table = config_dict["diag_table"]

    _validate_fortran_diagnostic_chunks(
        user_config.fortran_diagnostics,
        diag_table,
        timestep,
        run_duration,
        physics_output_interval,
    )
    _validate_python_diagnostic_chunks(
        user_config.diagnostics, initial_time, timestep, run_duration
    )


def _validate_fortran_diagnostic_chunks(
    diagnostics, diag_table, timestep, run_duration, physics_output_interval
):
    for diagnostic_config in diagnostics:
        name = os.path.splitext(diagnostic_config.name)[0]  # remove .zarr suffix
        time_chunk_size = diagnostic_config.chunks.get("time", None)
        matching_file_configs = [f for f in diag_table.file_configs if f.name == name]
        if matching_file_configs and time_chunk_size:
            file_config = matching_file_configs[0]
            for field_config in file_config.field_configs:
                num_output_timesteps = _get_fortran_num_output_timesteps(
                    file_config,
                    field_config,
                    physics_output_interval,
                    timestep,
                    run_duration,
                )
                _validate_time_chunks(
                    num_output_timesteps, time_chunk_size, file_config.name
                )


def _validate_python_diagnostic_chunks(
    diagnostics, initial_time, timestep, run_duration
):
    num_total_timesteps = int(run_duration / timestep)
    all_times = [initial_time + (n + 1) * timestep for n in range(num_total_timesteps)]
    for diagnostic_config in diagnostics:
        if "time" in diagnostic_config.chunks:
            time_container = diagnostic_config.times.time_container(initial_time)
            num_output_timesteps = sum(
                time_container.indicator(t) is not None for t in all_times
            )
            _validate_time_chunks(
                num_output_timesteps,
                diagnostic_config.chunks["time"],
                diagnostic_config.name,
            )


def _validate_time_chunks(time_dim_size: int, chunk_size: int, file_name: str):
    if time_dim_size % chunk_size != 0:
        raise ConfigValidationError(
            "Time dimension size must be a multiple of time chunk size. This was not "
            f"the case for an output in '{file_name}'."
        )


def _get_fortran_num_output_timesteps(
    file_config: fv3config.DiagFileConfig,
    field_config: fv3config.DiagFieldConfig,
    physics_output_interval: timedelta,
    timestep: timedelta,
    run_duration: timedelta,
) -> int:
    """Given configurations for Fortran diagnostic, return number of timesteps this
    diagnostic will have in output."""
    if field_config.module_name == "dynamics":
        # frequency is specified in diag_table
        if file_config.frequency == 0:
            interval = timestep
        elif file_config.frequency == -1:
            interval = run_duration
        else:
            interval = timedelta(**{file_config.frequency_units: file_config.frequency})
    else:
        # assuming from physics module, so frequency set in namelist
        interval = physics_output_interval

    if run_duration % interval == timedelta(0):
        return int(run_duration / interval)
    else:
        raise ConfigValidationError(
            "Run duration must be a multiple of the output interval for all "
            "diagnostics. This was not the case for output "
            f"'{field_config.output_name}' of file '{file_config.name}'."
        )
