import pytest
import numpy as np
import tensorflow as tf

from fv3fit.emulation.scoring import (
    score,
    score_multi_output,
    _append_rectified_cloud_if_available,
)


def _get_tensor():
    return tf.ones((10, 2))


def _get_ndarray():
    return np.ones((10, 2))


@pytest.fixture(params=[_get_tensor, _get_ndarray])
def arraylike_input(request):
    return request.param()


@pytest.fixture
def target(arraylike_input):
    return arraylike_input


@pytest.fixture
def prediction(arraylike_input):
    # add 1 to make bias, rmse, mse == 1.0
    return arraylike_input + 1


def test_score(target, prediction):

    scores, profiles = score(target, prediction)

    for v in scores.values():
        assert v == 1.0

    for v in profiles.values():
        np.testing.assert_array_equal(v, np.ones(2))


def test_score_multi_output_flat_scores(target, prediction):

    nout = 3
    targets = [target] * nout
    predictions = [prediction] * nout
    names = ["field{i}" for i in range(nout)]
    scores, profiles = score_multi_output(targets, predictions, names)

    for v in scores.values():
        assert v == 1.0

    for v in profiles.values():
        np.testing.assert_array_equal(v, np.ones(2))


def test__append_rectified_cloud_if_available():

    targets = [np.array([1, 1, 1])]
    predictions = [np.array([-1, 0, 1])]
    names = ["cloud_water_mixing_ratio_after_gscond"]

    _append_rectified_cloud_if_available(targets, predictions, names)

    for _list in [targets, predictions, names]:
        assert len(_list) == 2

    np.testing.assert_array_equal(predictions[1], np.array([0, 0, 1]))
    np.testing.assert_array_equal(targets[0], targets[1])
