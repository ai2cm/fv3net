from datetime import timedelta

import cftime
import pytest

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


@pytest.mark.parametrize(
    "times, chunks, should_validate",
    [
        (runtime.diagnostics.manager.TimeConfig(kind="every"), {}, True),
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
