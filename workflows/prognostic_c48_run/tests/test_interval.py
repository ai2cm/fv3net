import cftime
import datetime
import numpy as np
import pytest
import xarray as xr

from runtime.steppers.interval import IntervalStepper

da = xr.DataArray(np.zeros(10))


class MockStepper:
    def __init__(self):
        self.output_tendencies = {"output_tendency": da}
        self.output_diags = {"output_diags": da}
        self.output_state_updates = {"output_state_update": da}

    def __call__(self, time, state):
        return self.output_tendencies, self.output_diags, self.output_state_updates


START_TIME = cftime.DatetimeJulian(2020, 1, 1, 0, 0, 0)


@pytest.mark.parametrize(
    "interval, time_checks",
    [
        (
            3600,
            {
                START_TIME + datetime.timedelta(seconds=1800): False,
                START_TIME + datetime.timedelta(seconds=3600): True,
            },
        ),
        (
            10,
            {
                START_TIME + datetime.timedelta(seconds=5): False,
                START_TIME + datetime.timedelta(seconds=1800): True,
                START_TIME + datetime.timedelta(seconds=3600): True,
            },
        ),
    ],
)
def test_needs_update(interval, time_checks):
    interval_stepper = IntervalStepper(
        apply_interval_seconds=interval, stepper=MockStepper()
    )
    # The first time checked becomes the start time for the IntervalStepper
    assert interval_stepper._need_to_update(START_TIME) is False
    for time, needs_update in time_checks.items():
        assert interval_stepper._need_to_update(time) == needs_update


def test_initial_time():
    interval_stepper = IntervalStepper(
        apply_interval_seconds=3600, stepper=MockStepper()
    )
    t0 = cftime.DatetimeJulian(2020, 1, 1, 0, 0, 0)
    interval_stepper._need_to_update(t0)
    assert interval_stepper.start_time == t0


def test_call():
    call_times = [
        cftime.DatetimeJulian(2020, 1, 1, 0, 0, 0) + i * datetime.timedelta(seconds=900)
        for i in range(6)
    ]
    interval_stepper = IntervalStepper(
        apply_interval_seconds=3600, stepper=MockStepper()
    )
    state_updates = []
    for t in call_times:
        _, _, state_update = interval_stepper(t, state={})
        state_updates.append(state_update)
    assert "output_state_update" in state_updates.pop(4)
    assert all(len(update) == 0 for update in state_updates)
