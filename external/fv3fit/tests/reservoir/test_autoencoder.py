import numpy as np
from fv3fit.reservoir.autoencoder import build_concat_and_scale_only_autoencoder

a = np.array([[10, 10], [0.0, 0.0]])  # size=2, mean=5, std=5
b = np.array([[2.0, 2.0], [0.0, 0.0]])  # size=2, mean=1, std=1

test_inputs = [
    np.array([10, 10]).reshape(1, -1),
    np.array([3, 3]).reshape(1, -1),
]


def test_build_concat_and_scale_only_autoencoder_normalize():
    model = build_concat_and_scale_only_autoencoder(["a", "b"], [a, b])
    np.testing.assert_array_almost_equal(
        model.encode(test_inputs).reshape(-1), np.array([1, 1, 2, 2])
    )


def test_build_concat_and_scale_only_autoencoder_predict():
    model = build_concat_and_scale_only_autoencoder(["a", "b"], [a, b])
    np.testing.assert_array_almost_equal(model.predict(test_inputs), test_inputs)
