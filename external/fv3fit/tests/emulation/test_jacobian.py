import numpy as np
import tensorflow as tf

from fv3fit.emulation.jacobian import (
    get_jacobians,
    standardize_jacobians,
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
