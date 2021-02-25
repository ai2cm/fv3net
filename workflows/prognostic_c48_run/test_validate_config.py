from datetime import timedelta

import cftime
import pytest

import fv3config
import runtime
import validate_config


@pytest.fixture
def initial_time():
    return cftime.DatetimeJulian(2016, 8, 1)


@pytest.fixture
def timestep():
    return timedelta(minutes=15)


@pytest.fixture
def run_duration():
    return timedelta(hours=12)


@pytest.fixture
def physics_output_interval():
    return timedelta(hours=3)


@pytest.mark.parametrize(
    "times, chunks, should_validate",
    [
        (runtime.diagnostics.manager.TimeConfig(kind="every"), {}, True),
        (runtime.diagnostics.manager.TimeConfig(kind="every"), {"time": 12}, True),
        (runtime.diagnostics.manager.TimeConfig(kind="every"), {"time": 7}, False),
        (
            runtime.diagnostics.manager.TimeConfig(kind="interval", frequency=3600),
            {},
            True,
        ),
        (
            runtime.diagnostics.manager.TimeConfig(kind="interval", frequency=3600),
            {"time": 6},
            True,
        ),
        (
            runtime.diagnostics.manager.TimeConfig(kind="interval", frequency=3600),
            {"time": 12},
            True,
        ),
        (
            runtime.diagnostics.manager.TimeConfig(kind="interval", frequency=3600),
            {"time": 10},
            False,
        ),
        (
            runtime.diagnostics.manager.TimeConfig(
                kind="selected",
                times=["20160801.001500", "20160801.004500", "20160801.020000"],
            ),
            {"time": 2},
            False,
        ),
    ],
)
def test__validate_python_diagnostic_chunks(
    initial_time, timestep, run_duration, times, chunks, should_validate
):
    diagnostics = [
        runtime.DiagnosticFileConfig(
            "diags.zarr", variables=["air_temperature"], times=times, chunks=chunks,
        ),
    ]
    if should_validate:
        validate_config._validate_python_diagnostic_chunks(
            diagnostics, initial_time, timestep, run_duration
        )
    else:
        with pytest.raises(validate_config.ConfigValidationError):
            validate_config._validate_python_diagnostic_chunks(
                diagnostics, initial_time, timestep, run_duration
            )


def generate_diag_table(initial_time, module_name, frequency, frequency_units):
    field_config = fv3config.DiagFieldConfig(module_name, "name", "name")
    file_config = fv3config.DiagFileConfig(
        "filename", frequency, frequency_units, field_configs=[field_config]
    )
    return fv3config.DiagTable("name", initial_time, [file_config])


@pytest.mark.parametrize(
    "module_name, frequency, frequency_units, chunks, should_validate",
    [
        ("dynamics", 0, "hours", {}, True),
        ("dynamics", 0, "hours", {"time": 1}, True),
        ("dynamics", 0, "hours", {"time": 7}, False),
        ("dynamics", -1, "hours", {"time": 1}, True),
        ("dynamics", -1, "hours", {"time": 2}, False),
        ("dynamics", 1, "hours", {"time": 12}, True),
        ("dynamics", 1, "hours", {"time": 7}, False),
        ("dynamics", 7, "hours", {"time": 12}, False),
        ("gfsphys", 1, "hours", {"time": 8}, True),
        ("gfsphys", 1, "hours", {"time": 12}, False),
    ],
)
def test__validate_fortran_diagnostic_chunks(
    initial_time,
    timestep,
    run_duration,
    physics_output_interval,
    module_name,
    frequency,
    frequency_units,
    chunks,
    should_validate,
):
    diag_table = generate_diag_table(
        initial_time, module_name, frequency, frequency_units
    )
    diagnostics = [runtime.FortranFileConfig("filename.zarr", chunks=chunks)]
    if should_validate:
        validate_config._validate_fortran_diagnostic_chunks(
            diagnostics, diag_table, timestep, run_duration, physics_output_interval
        )
    else:
        with pytest.raises(validate_config.ConfigValidationError):
            validate_config._validate_fortran_diagnostic_chunks(
                diagnostics, diag_table, timestep, run_duration, physics_output_interval
            )
