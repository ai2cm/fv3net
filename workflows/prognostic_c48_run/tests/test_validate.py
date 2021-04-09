from datetime import timedelta
import cftime
import pytest
import runtime
from runtime.validate import _validate_time_chunks, ConfigValidationError


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
def test__validate_time_chunks(times, chunks, should_validate):
    initial_time = cftime.DatetimeJulian(2016, 8, 1)
    timestep = timedelta(minutes=15)
    run_duration = timedelta(hours=12)
    diag_file_config = runtime.config.DiagnosticFileConfig(
        "diags.zarr", variables=["air_temperature"], times=times, chunks=chunks,
    )

    if should_validate:
        _validate_time_chunks(diag_file_config, initial_time, timestep, run_duration)
    else:
        with pytest.raises(ConfigValidationError):
            _validate_time_chunks(
                diag_file_config, initial_time, timestep, run_duration
            )
