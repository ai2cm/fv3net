import fv3fit
import mock
import tensorflow as tf
import numpy as np
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
    n_batches = 5
    n_callbacks = 0
    config = fv3fit.TrainingLoopConfig()
    mock_model = mock.MagicMock(spec=tf.keras.Model)
    mock_Xy = []
    for _ in range(n_batches):
        mock_Xy.append(
            (mock.MagicMock(spec=np.ndarray), mock.MagicMock(spec=np.ndarray))
        )
    validation_data = (mock.MagicMock(spec=np.ndarray), mock.MagicMock(spec=np.ndarray))
    callbacks = tuple(mock.MagicMock for _ in range(n_callbacks))
    config.fit_loop(mock_model, mock_Xy, validation_data, callbacks)


@pytest.mark.parametrize("n_callbacks", [0, 1, 3])
@pytest.mark.parametrize("n_epochs", [0, 1, 3])
def test_fit_loop_calls_callbacks(n_callbacks, n_epochs):
    n_batches = 5
    config = fv3fit.TrainingLoopConfig(epochs=n_epochs)
    mock_model = mock.MagicMock(spec=tf.keras.Model)
    mock_history = mock.MagicMock(spec=tf.keras.callbacks.History)
    mock_model.fit.return_value = mock_history
    mock_Xy = []
    for _ in range(n_batches):
        mock_Xy.append(
            (mock.MagicMock(spec=np.ndarray), mock.MagicMock(spec=np.ndarray))
        )
    validation_data = (mock.MagicMock(spec=np.ndarray), mock.MagicMock(spec=np.ndarray))
    callbacks = tuple(mock.MagicMock() for _ in range(n_callbacks))
    config.fit_loop(mock_model, mock_Xy, validation_data, callbacks)
    for callback in callbacks:
        assert callback.call_count == n_epochs
        for i, call_args in enumerate(callback.call_args_list):
            assert call_args == mock.call(
                fv3fit.EpochResult(
                    epoch=i, history=tuple(mock_history for _ in range(n_batches))
                )
            )


@pytest.mark.parametrize(
    "n_epochs, n_batches", [(0, 1), (1, 1), (5, 1), (1, 5), (3, 4)]
)
def test_fit_loop_calls_fit(n_epochs, n_batches):
    n_batches = 5
    n_epochs = 1
    config = fv3fit.TrainingLoopConfig(epochs=n_epochs)
    mock_model = mock.MagicMock(spec=tf.keras.Model)
    mock_Xy = []
    for _ in range(n_batches):
        mock_Xy.append(
            (mock.MagicMock(spec=np.ndarray), mock.MagicMock(spec=np.ndarray))
        )
    validation_data = (mock.MagicMock(spec=np.ndarray), mock.MagicMock(spec=np.ndarray))
    config.fit_loop(mock_model, mock_Xy, validation_data)
    assert mock_model.fit.call_count == n_epochs * n_batches


@pytest.mark.parametrize("n_batches", [1, 3])
@pytest.mark.parametrize("n_epochs", [1, 5])
def test_fit_loop_calls_fit_with_data(n_batches, n_epochs):
    n_batches = 5
    n_epochs = 1
    config = fv3fit.TrainingLoopConfig(epochs=n_epochs)
    mock_model = mock.MagicMock(spec=tf.keras.Model)
    mock_Xy = []
    for _ in range(n_batches):
        mock_Xy.append(
            (mock.MagicMock(spec=np.ndarray), mock.MagicMock(spec=np.ndarray))
        )
    validation_data = (mock.MagicMock(spec=np.ndarray), mock.MagicMock(spec=np.ndarray))
    config.fit_loop(mock_model, mock_Xy, validation_data)
    assert mock_model.fit.call_count == len(mock_Xy)
    epoch_calls = []
    for i in range(n_epochs):
        epoch_calls.append(
            mock_model.fit.call_args_list[i * n_batches : (i + 1) * n_batches]
        )
    # check all data are passed as args on each epoch
    for call_args_list in epoch_calls:
        for X, y in mock_Xy:
            assert (
                mock.call(
                    X,
                    y,
                    batch_size=config.keras_batch_size,
                    validation_data=validation_data,
                )
                in call_args_list
            )


def test_fit_loop_calls_in_different_order_on_two_epochs():
    n_batches = 5
    n_epochs = 2
    config = fv3fit.TrainingLoopConfig(epochs=n_epochs)
    mock_model = mock.MagicMock(spec=tf.keras.Model)
    mock_Xy = []
    for _ in range(n_batches):
        mock_Xy.append(
            (mock.MagicMock(spec=np.ndarray), mock.MagicMock(spec=np.ndarray))
        )
    validation_data = (mock.MagicMock(spec=np.ndarray), mock.MagicMock(spec=np.ndarray))
    config.fit_loop(mock_model, mock_Xy, validation_data)
    assert mock_model.fit.call_count == len(mock_Xy) * n_epochs
    epoch_calls = []
    for i in range(n_epochs):
        epoch_calls.append(
            mock_model.fit.call_args_list[i * n_batches : (i + 1) * n_batches]
        )
    assert epoch_calls[0] != epoch_calls[1]


def test_fit_loop_calls_in_reproducible_order():
    n_batches = 5
    n_epochs = 2
    config = fv3fit.TrainingLoopConfig(epochs=n_epochs)
    first_mock_model = mock.MagicMock(spec=tf.keras.Model)
    second_mock_model = mock.MagicMock(spec=tf.keras.Model)
    mock_Xy = []
    for _ in range(n_batches):
        mock_Xy.append(
            (mock.MagicMock(spec=np.ndarray), mock.MagicMock(spec=np.ndarray))
        )
    validation_data = (mock.MagicMock(spec=np.ndarray), mock.MagicMock(spec=np.ndarray))
    fv3fit.set_random_seed(0)
    config.fit_loop(first_mock_model, mock_Xy, validation_data)
    fv3fit.set_random_seed(0)
    config.fit_loop(second_mock_model, mock_Xy, validation_data)
    assert first_mock_model.fit.call_args_list == second_mock_model.fit.call_args_list
