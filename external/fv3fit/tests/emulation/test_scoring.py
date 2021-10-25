import pytest
import numpy as np
import tensorflow as tf

from fv3fit.emulation import scoring
from fv3fit.emulation.scoring import score, score_single_output, score_multi_output


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


def test_score_single_output_flat_scores(target, prediction):

    scores, profiles = score_single_output(target, prediction, "field", rescale=False)

    for v in scores.values():
        assert v == 1.0

    for v in profiles.values():
        np.testing.assert_array_equal(v, np.ones(2))


def test_score_rescale(target, prediction, monkeypatch):

    fieldname = "field"
    monkeypatch.setitem(scoring.SCALE_VALUES, fieldname, 1 / 10)

    target *= 10
    prediction *= 10

    scores, profiles = score_single_output(target, prediction, fieldname, rescale=True)

    for v in scores.values():
        assert v == 1.0

    for v in profiles.values():
        np.testing.assert_array_equal(v, np.ones(2))


def test_score_multi_output_flat_scores(target, prediction):

    nout = 3
    targets = [target] * nout
    predictions = [prediction] * nout
    names = ["field{i}" for i in range(nout)]
    scores, profiles = score_multi_output(targets, predictions, names, rescale=False)

    for v in scores.values():
        assert v == 1.0

    for v in profiles.values():
        np.testing.assert_array_equal(v, np.ones(2))


def test_multi_out_score_rescale(target, prediction, monkeypatch):

    fieldname = "field"
    monkeypatch.setitem(scoring.SCALE_VALUES, fieldname, 1 / 10)

    target *= 10
    prediction *= 10

    scores, profiles = score_multi_output([target], [prediction], [fieldname], rescale=True)

    for v in scores.values():
        assert v == 1.0

    for v in profiles.values():
        np.testing.assert_array_equal(v, np.ones(2))
