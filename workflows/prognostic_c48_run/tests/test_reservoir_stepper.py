"""
Unit tests for the reservoir stepper.
"""

import pytest
import runtime.steppers.reservoir as reservoir
from runtime.steppers.reservoir import (
    _ReservoirStepper,
    _FiniteStateMachine,
    HybridReservoirConfig,
)
from datetime import datetime, timedelta
from unittest.mock import MagicMock


def test_reservoir_stepper_state():
    fsm = _FiniteStateMachine()

    assert fsm.increment_steps == 0

    for i in range(2):
        fsm.to_incremented()
        assert fsm._last_called == fsm.INCREMENT
        assert fsm.increment_steps == i + 1

    fsm.to_predicted()
    assert fsm._last_called == fsm.PREDICT
    assert fsm.increment_steps == 2


def test_reservior_stepper_state_call():

    fsm = _FiniteStateMachine()
    fsm(fsm.INCREMENT)
    fsm(fsm.PREDICT)
    with pytest.raises(ValueError):
        fsm("unknown_state")


def test_reservoir_stepper_state_predict_without_increment():
    fsm = _FiniteStateMachine()

    # Test that predict() raises a ValueError when increment() has not been called
    fsm._last_called = None
    with pytest.raises(ValueError):
        fsm.to_predicted()


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
    mock_model.hybrid_variables = ["a"]
    mock_model.rank_divider.overlap = 1
    mock_model.predict.return_value = {}

    return mock_model


def get_mock_ReservoirStepper(timestep=timedelta(minutes=10)):

    model = get_mock_reservoir_model()
    # Create a _ReservoirStepper object with mock objects
    stepper = _ReservoirStepper(
        model=model,
        reservoir_timestep=timedelta(minutes=10),
        synchronize_steps=2,
        state_machine=_FiniteStateMachine(init_time=datetime(1, 1, 1, 0, 0, 0)),
    )

    return stepper


@pytest.fixture(scope="function")
def patched_reservoir_module(monkeypatch):
    def append_halos_using_mpi(a, b):
        return {}

    monkeypatch.setattr(reservoir, "append_halos_using_mpi", append_halos_using_mpi)

    def open_rc_model(path):
        return get_mock_reservoir_model()

    monkeypatch.setattr(reservoir, "open_rc_model", open_rc_model)

    return reservoir


def test__ReservoirStepper__is_rc_update_step():
    # Test that the time check for a given TimeLoop time is correct

    stepper = get_mock_ReservoirStepper()

    time = stepper._state_machine.init_time
    assert stepper._is_rc_update_step(time)
    assert not stepper._is_rc_update_step(time + timedelta(minutes=5))
    assert stepper._is_rc_update_step(time + timedelta(minutes=10))
    assert stepper._is_rc_update_step(time + timedelta(hours=1, minutes=10))


def test__ReservoirStepper_model_sync(patched_reservoir_module):
    # Test that the model reset is called  when no completed steps and not otherwise
    # also check that completed sync steps is updated correctly

    stepper = get_mock_ReservoirStepper()
    mock_state = MockState(a=1)

    stepper.increment_reservoir(stepper.init_time, mock_state)
    stepper.model.reset_state.assert_called()
    assert stepper.completed_sync_steps == 1

    # shouldn't call reset again after it's been initialized
    stepper.increment_reservoir(stepper.init_time, mock_state)
    stepper.model.reset_state.assert_called_once()
    assert stepper.completed_sync_steps == 2


def test__ReservoirStepper_model_predict(patched_reservoir_module):
    # Test that the model predict is called  when completed steps and not otherwise
    # also check that completed sync steps is updated correctly

    stepper = get_mock_ReservoirStepper()
    mock_state = MockState(a=1)

    # no call to predict when at or below required number of sync steps
    for i in range(stepper.synchronize_steps):
        stepper.increment_reservoir(stepper.init_time, mock_state)
        stepper.hybrid_predict(stepper.init_time, mock_state)
        stepper.model.predict.assert_not_called()

    # call to predict when past synchronization period
    stepper.increment_reservoir(stepper.init_time, mock_state)
    stepper.hybrid_predict(stepper.init_time, mock_state)
    stepper.model.predict.assert_called_once()


def test_get_reservoir_steppers(patched_reservoir_module):

    config = HybridReservoirConfig("model", 0, reservoir_timestep="10m")
    incrementer, predictor = reservoir.get_reservoir_steppers(config)

    # Check that both steppers share model and state machine objects
    assert incrementer.model is predictor.model
    assert incrementer._state_machine is predictor._state_machine

    # check that call methods point to correct methods
    time = datetime(1, 1, 1, 0, 0, 0)
    state = MockState(a=1)
    incrementer(time, state)
    incrementer.model.increment_state.assert_called()
    predictor(time, state)
    predictor.model.predict.assert_called()


def test_reservoir_steppers_state_machine_constraint(patched_reservoir_module):

    config = HybridReservoirConfig("model", 0, reservoir_timestep="10m")
    incrementer, predictor = reservoir.get_reservoir_steppers(config)

    # check that steppers respect state machine limit
    time = datetime(1, 1, 1, 0, 0, 0)
    state = MockState(a=1)
    incrementer(time, state)
    incrementer(time, state)
    predictor(time, state)
    with pytest.raises(ValueError):
        predictor(time, state)
