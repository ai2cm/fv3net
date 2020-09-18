from cftime import DatetimeJulian as datetime
from unittest.mock import Mock

import pytest
import xarray as xr

from runtime import diagnostics
from runtime.diagnostics import DiagnosticFile, All


@pytest.mark.parametrize(
    "time_stamp",
    [
        pytest.param("20160801.000000", id="full time stamp"),
        pytest.param("20160801.00000", id="time stamp with one less 0"),
    ],
)
def test_SelectedTimes(time_stamp):
    times = diagnostics.SelectedTimes([time_stamp])
    time = datetime(year=2016, month=8, day=1, hour=0, minute=0, second=0)
    assert time in times


def test_SelectedTimes_not_in_list():
    times = diagnostics.SelectedTimes(["20160801.000000"])
    time = datetime(year=2016, month=8, day=1, hour=0, minute=0, second=1)
    assert time not in times


august_1 = datetime(year=2016, month=8, day=1, hour=0, minute=0)
august_2 = datetime(year=2016, month=8, day=2, hour=0, minute=0)


@pytest.mark.parametrize(
    "frequency, time, initial_time, expected",
    [
        (900, datetime(year=2016, month=8, day=1, hour=0, minute=15), august_1, True),
        (900, datetime(year=2016, month=8, day=1, hour=0, minute=16), august_1, False),
        (900, datetime(year=2016, month=8, day=1, hour=12, minute=45), august_1, True),
        (86400, datetime(year=2016, month=8, day=1, hour=0, minute=0), august_1, True),
        (86400, datetime(year=2016, month=8, day=2, hour=0, minute=0), august_2, True),
        pytest.param(
            5 * 60 * 60,
            datetime(year=2016, month=8, day=2),
            datetime(year=2016, month=8, day=1),
            False,
            id="5hourlyFalse",
        ),
        pytest.param(
            5 * 60 * 60,
            datetime(year=2016, month=8, day=2, hour=1),
            datetime(year=2016, month=8, day=1),
            True,
            id="5hourlyTrue",
        ),
    ],
)
def test_IntervalTimes(frequency, time, initial_time, expected):
    times = diagnostics.IntervalTimes(frequency, initial_time)
    assert (time in times) == expected


def test_DiagnosticFile_time_selection():
    # mock the input data
    t1 = datetime(year=2016, month=8, day=1, hour=0, minute=15)
    t2 = datetime(year=2016, month=8, day=1, hour=0, minute=16)

    monitor = Mock()

    # observe a few times
    diag_file = DiagnosticFile(monitor, times=[t1], variables=All())
    diag_file.observe(t1, {})
    diag_file.observe(t2, {})
    monitor.store.assert_called_once()


def test_DiagnosticFile_variable_selection():

    data_vars = {"a": (["x"], [1.0]), "b": (["x"], [2.0])}
    dataset = xr.Dataset(data_vars)
    diagnostics = {key: dataset[key] for key in dataset}

    class VariableCheckingMonitor:
        def store(self, state):
            assert "time" in state
            assert "a" in state
            assert "b" not in state

    monitor = VariableCheckingMonitor()

    # observe a few times
    diag_file = DiagnosticFile(monitor, times=All(), variables=["a"])
    diag_file.observe(None, diagnostics)


@pytest.mark.parametrize(
    "attrs, expected_units", [({}, "unknown"), ({"units": "zannyunits"}, "zannyunits")]
)
def test_DiagnosticFile_variable_units(attrs, expected_units):
    data_vars = {"a": (["x"], [1.0], attrs)}
    dataset = xr.Dataset(data_vars)
    diagnostics = {key: dataset[key] for key in dataset}

    class UnitCheckingMonitor:
        def store(self, state):
            assert state["a"].units == expected_units

    monitor = UnitCheckingMonitor()

    # observe a few times
    diag_file = DiagnosticFile(monitor, times=All(), variables=All())
    diag_file.observe(None, diagnostics)
