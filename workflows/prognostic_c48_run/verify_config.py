from datetime import timedelta
from typing import Mapping

import dacite
import fv3config
import fsspec
from runtime.config import UserConfig


def verify_config(config_dict: Mapping):
    _verify_chunks(config_dict)


def _verify_chunks(config_dict: Mapping):
    user_config = dacite.from_dict(UserConfig, config_dict)
    run_duration = fv3config.get_run_duration(config_dict)
    if isinstance(config["diag_table"], str):
        # assume diag_table item is path to a file
        with fsspec.open(config["diag_table"]) as f:
            diag_table = fv3config.DiagTable.from_str(f.read())
    else:
        diag_table = config["diag_table"]

    _verify_fortran_diagnostic_chunks(
        diag_table, config_dict["namelist"], user_config.fortran_diagnostics,
    )
    _verify_python_diagnostic_chunks(user_config.diagnostics)


def _verify_fortran_diagnostic_chunks(diag_table, namelist, diagnostics):
    chunks = runtime.get_chunks(diagnostics)
    physics_diagnostic_frequency = timedelta(hours=namelist["atmos_model_nml"]["fhout"])
    timestep = timedelta(seconds=namelist["coupler_nml"]["dt_atmos"])
    for file_config in diag_table.file_configs:
        zarr_name = f"{file_config.name}.zarr"
        if zarr_name in chunks:
            time_chunk_size = chunks[zarr_name]["time"]
            for field_config in file_config.field_configs:
                frequency = _get_frequency_of_field(
                    field_config, physics_diagnostic_frequency, timestep, run_duration
                )
                _verify_time_chunks(run_duration, frequency, time_chunk_size)


def _verify_python_diagnostic_chunks(diagnostics):
    chunks = runtime.get_chunks(diagnostics)


def _verify_time_chunks(run_duration: timedelta, frequency: timedelta, chunk_size: int):
    # ensure output frequency evenly divides run length
    assert run_duration % frequency == timedelta(0)
    # ensure chunk size evenly divides size of time dimension
    time_dim_size = int(duration / frequency)
    assert time_dim_size % chunk_size == 0


def _get_frequency_of_field(
    config: fv3config.DiagFieldConfig,
    physics_frequency: timedelta,
    timestep: timedelta,
    run_duration: timedelta,
) -> timedelta:
    if config.module_name == "dynamics":
        # frequency is specified in diag_table
        if config.frequency == 0:
            return timestep
        elif config.frequency == -1:
            return run_duration
        else:
            return timedelta(**{config.frequency_units: config.frequency})
    else:
        # assuming from physics module, so frequency set in namelist
        return physics_frequency
