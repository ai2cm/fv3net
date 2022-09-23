import cftime
import datetime
import numpy as np
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


def test_needs_update():
    interval_stepper = IntervalStepper(
        apply_interval_seconds=3600, stepper=MockStepper()
    )
    assert (
        interval_stepper._need_to_update(cftime.DatetimeJulian(2020, 1, 1, 0, 0, 0))
        is False
    )
    half_hour_check = interval_stepper._need_to_update(
        cftime.DatetimeJulian(2020, 1, 1, 0, 30, 0)
    )
    assert half_hour_check is False
    hour_check = interval_stepper._need_to_update(
        cftime.DatetimeJulian(2020, 1, 1, 1, 0, 0)
    )
    assert hour_check is True


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
