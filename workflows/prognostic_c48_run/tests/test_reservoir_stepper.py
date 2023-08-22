"""
Unit tests for the reservoir stepper.
"""

import numpy as np
import xarray as xr
import pytest
import runtime.steppers.reservoir as reservoir
from runtime.steppers.reservoir import (
    ReservoirIncrementOnlyStepper,
    ReservoirPredictStepper,
    _FiniteStateMachine,
    IntervalAveragedTimes,
    TimeAverageInputs,
    ReservoirConfig,
)
from datetime import datetime, timedelta
from unittest.mock import MagicMock


def test_reservoir_stepper_state():
    fsm = _FiniteStateMachine()

    assert fsm.completed_increments == 0

    for i in range(2):
        fsm.to_incremented()
        assert fsm._last_called == fsm.INCREMENT
        assert fsm.completed_increments == i + 1

    fsm.to_predicted()
    assert fsm._last_called == fsm.PREDICT
    assert fsm.completed_increments == 2


def test_state_machine_call():

    fsm = _FiniteStateMachine()
    fsm(fsm.INCREMENT)
    fsm(fsm.PREDICT)
    with pytest.raises(ValueError):
        fsm("unknown_state")


def test_state_machine_predict_without_increment():
    fsm = _FiniteStateMachine()

    # Test that predict() raises a ValueError when increment() has not been called
    fsm._last_called = None
    with pytest.raises(ValueError):
        fsm.to_predicted()


def test_input_averager_increment_running_average():
    averager = TimeAverageInputs(["a"])
    data = xr.Dataset({"a": xr.DataArray(np.ones(1), dims=["x"])})

    averager.increment_running_average(data)
    xr.testing.assert_equal(averager._running_total["a"], data["a"])

    # add an extra variable to check that it's ignored
    data["b"] = np.ones(1) * 5
    averager.increment_running_average(data)
    assert len(averager._running_total) == 1
    xr.testing.assert_equal(averager._running_total["a"], data["a"] * 2)

    with pytest.raises(KeyError):
        averager.increment_running_average({})


def test_input_averager_get_averages():
    averager = TimeAverageInputs(["a"])
    data = xr.Dataset({"a": xr.DataArray(np.ones(1), dims=["x"])})

    with pytest.raises(ValueError):
        averager.get_averages()

    # single increment
    averager.increment_running_average(data)
    result = averager.get_averages()
    assert len(result) == 1
    xr.testing.assert_equal(result["a"], data["a"])

    # multiple increments
    averager.increment_running_average(data)
    averager.increment_running_average(data * 2)
    averager.increment_running_average(data * 3)
    result = averager.get_averages()
    xr.testing.assert_equal(result["a"], xr.DataArray(np.ones(1) * 2.0, dims=["x"]))


def test_input_averager_no_variables():
    averager = TimeAverageInputs([])
    averager.increment_running_average({})
    averager.get_averages()


class MockState(dict):
    # Mocking the dataset variable filtering for xr.Dataset
    def __getitem__(self, key):
        d = dict(**self)
        if isinstance(key, list):
            return {k: d[k] for k in key}
        else:
            return d[key]


def get_mock_reservoir_model():
    mock_model = MagicMock()
    mock_model.input_variables = ["a"]
    mock_model.output_variables = ["a"]
    mock_model.model.input_variables = ["a"]
    mock_model.model.hybrid_variables = ["a"]
    mock_model.input_overlap = 1
    out_data = xr.DataArray(np.ones(1), dims=["x"])
    mock_model.predict.return_value = {"a": out_data}

    return mock_model


def get_mock_ReservoirSteppers():

    model = get_mock_reservoir_model()
    state_machine = _FiniteStateMachine()

    time_checker = IntervalAveragedTimes(
        timedelta(minutes=10), datetime(1, 1, 1, 0, 0, 0)
    )

    # Create a _ReservoirStepper object with mock objects
    incrementer = ReservoirIncrementOnlyStepper(
        model=model,
        time_checker=time_checker,
        synchronize_steps=2,
        state_machine=state_machine,
    )

    predictor = ReservoirPredictStepper(
        model=model,
        time_checker=time_checker,
        synchronize_steps=2,
        state_machine=state_machine,
    )

    return incrementer, predictor


@pytest.fixture(scope="function")
def patched_reservoir_module(monkeypatch):
    def append_halos_using_mpi(inputs, nhalo):
        return inputs

    monkeypatch.setattr(reservoir, "append_halos_using_mpi", append_halos_using_mpi)

    def open_rc_model(path):
        return get_mock_reservoir_model()

    monkeypatch.setattr(reservoir, "open_rc_model", open_rc_model)

    return reservoir


def test__ReservoirStepper__is_rc_update_step():
    # Test that the time check for a given TimeLoop time is correct

    stepper, _ = get_mock_ReservoirSteppers()

    time = stepper.initial_time
    assert stepper._is_rc_update_step(time)
    assert not stepper._is_rc_update_step(time + timedelta(minutes=5))
    assert stepper._is_rc_update_step(time + timedelta(minutes=10))
    assert stepper._is_rc_update_step(time + timedelta(hours=1, minutes=10))


def test__ReservoirStepper_model_sync(patched_reservoir_module):
    # Test that the model reset is called  when no completed steps and not otherwise
    # also check that completed sync steps is updated correctly

    incrementer, _ = get_mock_ReservoirSteppers()
    mock_state = MockState(a=1)

    incrementer.increment_reservoir(mock_state)
    incrementer.model.reset_state.assert_called()
    assert incrementer.completed_sync_steps == 1

    # shouldn't call reset again after it's been initialized
    incrementer.increment_reservoir(mock_state)
    incrementer.model.reset_state.assert_called_once()
    assert incrementer.completed_sync_steps == 2


def test__ReservoirStepper_model_predict(patched_reservoir_module):
    # Test that the model predict is called  when completed steps and not otherwise
    # also check that completed sync steps is updated correctly

    incrementer, predictor = get_mock_ReservoirSteppers()
    mock_state = MockState(a=xr.DataArray(np.ones(1), dims=["x"]))

    # no call to predict when at or below required number of sync steps
    for i in range(incrementer.synchronize_steps):
        incrementer.increment_reservoir(mock_state)
        predictor.predict(mock_state, mock_state)
        predictor.model.predict.assert_not_called()

    # call to predict when past synchronization period
    incrementer.increment_reservoir(mock_state)
    predictor.predict(mock_state, mock_state)
    predictor.model.predict.assert_called_once()


def test_get_reservoir_steppers(patched_reservoir_module):

    config = ReservoirConfig({0: "model"}, 0, reservoir_timestep="10m")
    time = datetime(2020, 1, 1, 0, 0, 0)
    incrementer, predictor = reservoir.get_reservoir_steppers(config, 0, time)

    # Check that both steppers share model and state machine objects
    assert incrementer.model is predictor.model
    assert incrementer._state_machine is predictor._state_machine
    assert incrementer.input_averager.variables == ["a"]
    assert predictor.input_averager.variables == ["a"]

    # check that call methods point to correct methods
    state = MockState(a=xr.DataArray(np.ones(1), dims=["x"]))
    incrementer(time, state)
    incrementer.model.increment_state.assert_called()
    predictor(time, state)
    predictor.model.predict.assert_called()


def test_reservoir_steppers_state_machine_constraint(patched_reservoir_module):

    config = ReservoirConfig({0: "model"}, 0, reservoir_timestep="10m")
    time = datetime(2020, 1, 1, 0, 0, 0)
    incrementer, predictor = reservoir.get_reservoir_steppers(config, 0, time)

    # check that steppers respect state machine limit
    state = MockState(a=xr.DataArray(np.ones(1), dims=["x"]))
    incrementer(time, state)
    incrementer(time, state)
    predictor(time, state)
    with pytest.raises(ValueError):
        predictor(time, state)


def test_model_paths_and_rank_index_mismatch_on_load():
    config = ReservoirConfig({1: "model"}, 0, reservoir_timestep="10m")
    with pytest.raises(KeyError):
        reservoir.get_reservoir_steppers(config, 1, datetime(2020, 1, 1))
