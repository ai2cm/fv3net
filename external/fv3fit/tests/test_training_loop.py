from typing import Tuple
import fv3fit
from unittest import mock
from fv3fit.keras._models.shared.training_loop import (
    _shuffle_batched_tfdataset,
    sequence_size,
)
import tensorflow as tf
import pytest
import contextlib
import numpy as np


def test_mock_in_sequence():
    """
    make sure that `mock in sequence` is True if and
    only if that instance is in the sequence
    """
    m = mock.MagicMock()
    assert m in [m]
    assert m not in [mock.MagicMock()]


@contextlib.contextmanager
def mock_tfdataset_to_tensor_sequence():
    tfdataset_to_tensor_sequence_mock = mock.MagicMock()
    tfdataset_to_tensor_sequence_mock.return_value = [
        mock.MagicMock(),
        mock.MagicMock(),
    ]
    with mock.patch(
        "fv3fit.keras._models.shared.training_loop._tfdataset_to_tensor_sequence",
        new=tfdataset_to_tensor_sequence_mock,
    ) as m:
        yield m


def test_fit_loop():
    n_callbacks = 0
    config = fv3fit.TrainingLoopConfig()
    mock_model = mock.MagicMock(spec=tf.keras.Model)
    mock_Xy = mock.MagicMock(spec=tf.data.Dataset)
    validation_data = mock.MagicMock(spec=tf.data.Dataset)
    callbacks = tuple(mock.MagicMock for _ in range(n_callbacks))
    with mock_tfdataset_to_tensor_sequence():
        config.fit_loop(mock_model, mock_Xy, validation_data, callbacks)


@pytest.mark.parametrize("n_epochs", [0, 1, 5])
def test_fit_loop_calls_fit(n_epochs):
    n_epochs = 1
    config = fv3fit.TrainingLoopConfig(epochs=n_epochs)
    mock_model = mock.MagicMock(spec=tf.keras.Model)
    mock_Xy = mock.MagicMock(spec=tf.data.Dataset)
    validation_data = mock.MagicMock(spec=tf.data.Dataset)
    with mock_tfdataset_to_tensor_sequence():
        config.fit_loop(mock_model, mock_Xy, validation_data)
    assert mock_model.fit.call_count == 1


def get_monotonic_data(
    n_samples: int, batch_size: int
) -> Tuple[tf.data.Dataset, np.ndarray]:
    array_x = np.arange(n_samples)
    x = tf.data.Dataset.from_tensor_slices(array_x).batch(batch_size)
    y = tf.data.Dataset.from_tensor_slices(array_x).batch(batch_size)
    data = tf.data.Dataset.zip((x, y))
    return data, array_x


def assert_data_is_shuffled(Xy, array_x, n_batches):
    n_samples = sequence_size(Xy)
    shuffled_x, shuffled_y = next(iter(Xy.batch(n_samples)))
    np.testing.assert_array_equal(shuffled_x, shuffled_y)
    assert not np.all(shuffled_x == np.arange(n_samples))
    # this code is here for manual checking, since it's hard to assert that something
    # is shuffled
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.hist2d(array_x, shuffled_x, bins=n_batches)
    # plt.xlabel("output index")
    # plt.ylabel("input index")
    # plt.show()

    # check that there is some inter-batch shuffling
    hist, _, _ = np.histogram2d(array_x, shuffled_x, bins=n_batches)
    assert np.sum((hist > 0).astype(np.int)) > n_batches * 5
    # check that input batch is uncorrelated with output batch
    assert (np.corrcoef(array_x, shuffled_x)[0, 1] ** 2) < 0.1


@pytest.mark.slow
def test__shuffle_batched_tfdataset():
    batch_size = 100
    n_samples = 20_000
    data, array_x = get_monotonic_data(n_samples, batch_size)
    n_batches = n_samples // batch_size

    shuffled_data = _shuffle_batched_tfdataset(data, sample_buffer_size=batch_size)
    assert_data_is_shuffled(shuffled_data, array_x=array_x, n_batches=n_batches)


@pytest.mark.slow
def test_fit_loop_shuffles_batches():
    # for test, batch_shuffle_buffer_size should fit all batches
    # and shuffle_buffer_size should not
    # in_memory must be False because keras's model.fit handle shuffling for True
    batch_size = 100
    n_samples = 20_000
    data, array_x = get_monotonic_data(n_samples, batch_size)
    mock_model = mock.MagicMock()
    n_batches = n_samples // batch_size
    config = fv3fit.TrainingLoopConfig(shuffle_buffer_size=batch_size, in_memory=False)
    mock_model = mock.MagicMock()

    config.fit_loop(model=mock_model, Xy=data, validation_data=None, callbacks=())
    args = mock_model.fit.call_args_list[0]
    Xy = args.args[0].unbatch()
    assert_data_is_shuffled(Xy, array_x=array_x, n_batches=n_batches)
