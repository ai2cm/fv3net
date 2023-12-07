import numpy as np
import pytest
from fv3fit.reservoir.transformers.autoencoder import (
    Autoencoder,
    build_concat_and_scale_only_autoencoder,
)

a = np.array([[10, 10], [0.0, 0.0]])  # size=2, mean=5, std=5
b = np.array([[2.0, 2.0], [0.0, 0.0]])  # size=2, mean=1, std=1

test_inputs = [
    np.array([10, 10]).reshape(1, -1),
    np.array([3, 3]).reshape(1, -1),
]


def test_decode_single_output_returns_list():
    model = build_concat_and_scale_only_autoencoder(["a"], [a])
    encoded = model.encode([test_inputs[0]])
    decoded = model.decode(encoded)
    assert isinstance(decoded, list)
    assert len(decoded) == 1


def test_build_concat_and_scale_only_autoencoder_normalize():
    model = build_concat_and_scale_only_autoencoder(["a", "b"], [a, b])
    np.testing.assert_array_almost_equal(
        model.encode(test_inputs).numpy().reshape(-1), np.array([1, 1, 2, 2])
    )


def test_build_concat_and_scale_only_autoencoder_predict():
    model = build_concat_and_scale_only_autoencoder(["a", "b"], [a, b])
    np.testing.assert_array_almost_equal(model.predict(test_inputs), test_inputs)


def test_concat_and_scale_only_autoencoder_dump_load(tmpdir):
    output_path = f"{str(tmpdir)}/model"
    model = build_concat_and_scale_only_autoencoder(["a", "b"], [a, b])
    encoded = model.encode(test_inputs)
    model.dump(output_path)
    loaded_model = Autoencoder.load(output_path)
    loaded_encoded = loaded_model.encode(test_inputs)
    np.testing.assert_array_equal(encoded, loaded_encoded)


def test_autoencoder_sample_dim_handling():
    model = build_concat_and_scale_only_autoencoder(["a", "b"], [a, b])
    res = model.encode([np.array([10, 10]), np.array([3, 3])])
    assert res.shape == (1, 4)

    res = model.encode([np.array([[10, 10]]), np.array([[3, 3]])])
    assert res.shape == (1, 4)


def test_autoencoder_more_than_2d():
    model = build_concat_and_scale_only_autoencoder(["a", "b"], [a, b])
    with pytest.raises(ValueError):
        model.encode([np.array([[[10, 10]]]), np.array([[[3, 3]]])])
