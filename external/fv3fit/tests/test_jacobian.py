import numpy as np
import tensorflow as tf
import pytest


from fv3fit._shared.jacobian import (
    get_jacobians,
    standardize_jacobians,
    standardize_jacobians_by_vertical_level,
)


def test_jacobians():
    n = 5
    sample = {
        "a": tf.random.normal((10000, n)),
        "b": tf.random.normal((10000, n)),
    }

    sample["field"] = sample["a"]

    profiles = {
        name: tf.reduce_mean(sample[name], axis=0, keepdims=True) for name in ["a", "b"]
    }

    def model(x):
        # no effect from 'b'
        return {"field": x["a"] + x["b"] * 0}

    jacobians = get_jacobians(model, profiles)
    normalized = standardize_jacobians(jacobians, sample)

    assert set(normalized) == {"field"}
    assert set(normalized["field"]) == {"a", "b"}

    np.testing.assert_array_almost_equal(normalized["field"]["a"], np.eye(n))
    np.testing.assert_array_almost_equal(normalized["field"]["b"], np.zeros((n, n)))


@pytest.mark.parametrize(
    "units, expected_diagonal", [("dimensionless", 1), ("output", 10000)]
)
def test_jacobians_vertical_level_stds(units, expected_diagonal):
    n = 5
    sample = {
        "input": tf.random.normal((10000, n), stddev=10000),
        "output": tf.random.normal((10000, n), stddev=0.1),
    }
    sample["field"] = sample["input"]

    profiles = {
        name: tf.reduce_mean(sample[name], axis=0, keepdims=True)
        for name in ["input", "output"]
    }

    def model(x):
        # no effect from 'b'
        return {"field": x["input"] + x["output"] * 0}

    jacobians = get_jacobians(model, profiles)
    normalized = standardize_jacobians_by_vertical_level(jacobians, sample, units)

    assert set(normalized) == {"field"}
    assert set(normalized["field"]) == {"input", "output"}

    # check that the matrix is diagnonal with values of expected_diagonal
    np.testing.assert_array_almost_equal(
        normalized["field"]["input"] / expected_diagonal, np.eye(n), decimal=2
    )
    np.testing.assert_array_almost_equal(
        normalized["field"]["output"], np.zeros((n, n))
    )
