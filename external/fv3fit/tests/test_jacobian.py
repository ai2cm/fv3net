import numpy as np
import tensorflow as tf

from fv3fit.keras.jacobian import (
    get_jacobians,
    nondimensionalize_jacobians,
)
from fv3fit.emulation.layers.normalization import standard_deviation_all_features


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

    std_factors = {
        name: np.array(float(standard_deviation_all_features(data)))
        for name, data in sample.items()
    }
    normalized = nondimensionalize_jacobians(jacobians, std_factors)

    assert set(normalized) == {"field"}
    assert set(normalized["field"]) == {"a", "b"}

    np.testing.assert_array_almost_equal(normalized["field"]["a"], np.eye(n))
    np.testing.assert_array_almost_equal(normalized["field"]["b"], np.zeros((n, n)))


def test_jacobians_vertical_level_stds():
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
    std_factors = {name: np.std(data, axis=0) for name, data in sample.items()}
    normalized = nondimensionalize_jacobians(jacobians, std_factors)

    assert set(normalized) == {"field"}
    assert set(normalized["field"]) == {"input", "output"}

    # check that the matrix is diagnonal with values of expected_diagonal
    np.testing.assert_array_almost_equal(normalized["field"]["input"], np.eye(n))
    np.testing.assert_array_almost_equal(
        normalized["field"]["output"], np.zeros((n, n))
    )
