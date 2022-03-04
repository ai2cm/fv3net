import fv3fit
from unittest import mock
from fv3fit.keras._models.shared.training_loop import EpochCallback
import tensorflow as tf
import pytest


def test_mock_in_sequence():
    """
    make sure that `mock in sequence` is True if and
    only if that instance is in the sequence
    """
    m = mock.MagicMock()
    assert m in [m]
    assert m not in [mock.MagicMock()]


def test_fit_loop():
    n_callbacks = 0
    config = fv3fit.TrainingLoopConfig()
    mock_model = mock.MagicMock(spec=tf.keras.Model)
    mock_Xy = mock.MagicMock(spec=tf.data.Dataset)
    validation_data = mock.MagicMock(spec=tf.data.Dataset)
    callbacks = tuple(mock.MagicMock for _ in range(n_callbacks))
    config.fit_loop(mock_model, mock_Xy, validation_data, callbacks)


@pytest.mark.parametrize("n_callbacks", [0, 1, 3])
@pytest.mark.parametrize("n_epochs", [0, 1, 3])
def test_fit_loop_passes_callbacks(n_callbacks, n_epochs):
    n_batches = 5
    config = fv3fit.TrainingLoopConfig(epochs=n_epochs)
    mock_model = mock.MagicMock(spec=tf.keras.Model)
    mock_history = mock.MagicMock(spec=tf.keras.callbacks.History)
    mock_model.fit.return_value = mock_history
    mock_Xy = mock.MagicMock(spec=tf.data.Dataset)
    validation_data = mock.MagicMock(spec=tf.data.Dataset)
    callbacks = tuple(mock.MagicMock() for _ in range(n_callbacks))
    config.fit_loop(mock_model, mock_Xy, validation_data, callbacks)
    assert mock_model.fit.called
    args = mock_model.fit.call_args
    for wrapper in args.kwargs["callbacks"]:
        assert isinstance(wrapper, EpochCallback)
    passed_callbacks = [wrapper._callback for wrapper in args.kwargs["callbacks"]]
    assert set(passed_callbacks) == set(callbacks)
    for callback in callbacks:
        for i, call_args in enumerate(callback.call_args_list):
            assert call_args == mock.call(
                fv3fit.EpochResult(
                    epoch=i, history=tuple(mock_history for _ in range(n_batches))
                )
            )


@pytest.mark.parametrize("n_epochs", [0, 1, 5])
def test_fit_loop_calls_fit(n_epochs):
    n_epochs = 1
    config = fv3fit.TrainingLoopConfig(epochs=n_epochs)
    mock_model = mock.MagicMock(spec=tf.keras.Model)
    mock_Xy = mock.MagicMock(spec=tf.data.Dataset)
    validation_data = mock.MagicMock(spec=tf.data.Dataset)
    config.fit_loop(mock_model, mock_Xy, validation_data)
    assert mock_model.fit.call_count == 1
