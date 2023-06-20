from fv3fit.reservoir.transformers.sk_transformer import SkTransformer
import numpy as np
import pytest
import os
from sklearn.preprocessing import StandardScaler


class DummyTransformer:
    # tranform pads extra features on either end,
    # inverse transform removes these
    def __init__(self, extra_embedded_dims: int = 5):
        self.extra_embedded_dims = extra_embedded_dims

    def transform(self, X: np.ndarray):
        nonpadded = ((0, 0) for d in range(X.ndim - 1))
        npad = (self.extra_embedded_dims, self.extra_embedded_dims)
        return np.pad(
            X, pad_width=(*nonpadded, npad), mode="constant", constant_values=10
        )

    def inverse_transform(self, X: np.ndarray):
        sl = [slice(None)] * X.ndim
        sl[-1] = slice(self.extra_embedded_dims, -self.extra_embedded_dims)
        return X[tuple(sl)]


def test_sktransformer_roundtrip():
    output_dim = 10
    n_vars = 3
    n_samples = 9

    scaler = StandardScaler()
    scaler.fit(np.random.rand(5, n_vars * output_dim))
    transformer = transformer = DummyTransformer(extra_embedded_dims=5)
    sktransformer = SkTransformer(transformer, scaler, enforce_positive_outputs=False)

    x = [np.random.rand(n_samples, output_dim) for i in range(n_vars)]
    outputs = sktransformer.predict(x)

    np.testing.assert_allclose(
        np.concatenate(outputs, axis=-1), np.concatenate(x, axis=-1)
    )


def test_sktransformer_enforce_positive():
    output_dim = 10
    n_vars = 2
    n_samples = 9

    scaler = StandardScaler()
    scaler.fit(np.random.rand(5, n_vars * output_dim))
    transformer = transformer = DummyTransformer(extra_embedded_dims=5)
    sktransformer = SkTransformer(transformer, scaler, enforce_positive_outputs=False)
    sktransformer_enforce_positive = SkTransformer(
        transformer, scaler, enforce_positive_outputs=True
    )

    x = [np.ones((n_samples, output_dim)), np.ones((n_samples, output_dim)) * -1]
    outputs = sktransformer.predict(x)
    outputs_enforce_positive = sktransformer_enforce_positive.predict(x)

    np.testing.assert_allclose(outputs[0], outputs_enforce_positive[0])
    assert np.all(outputs_enforce_positive[1] == 0)


def test_sktransformer_dump_load(tmpdir):
    output_dim = 10
    n_vars = 2
    n_samples = 9

    scaler = StandardScaler()
    scaler.fit(np.random.rand(5, n_vars * output_dim))
    transformer = transformer = DummyTransformer(extra_embedded_dims=5)
    sktransformer = SkTransformer(transformer, scaler, enforce_positive_outputs=True)

    x = [np.random.rand(n_samples, output_dim) for i in range(n_vars)]
    prediction = sktransformer.predict(x)

    path = os.path.join(str(tmpdir), "sktransformer")
    sktransformer.dump(path)

    loaded_sktransformer = SkTransformer.load(path)
    loaded_prediction = loaded_sktransformer.predict(x)

    for var, loaded_var in zip(prediction, loaded_prediction):
        np.testing.assert_allclose(var, loaded_var)


def test_sktransformer_no_sample_dim():
    output_dim = 10
    n_vars = 3

    scaler = StandardScaler()
    scaler.fit(np.random.rand(5, n_vars * output_dim))
    transformer = transformer = DummyTransformer(extra_embedded_dims=5)
    sktransformer = SkTransformer(transformer, scaler, enforce_positive_outputs=False)

    x = [np.random.rand(output_dim) for i in range(n_vars)]
    sktransformer.predict(x)


def test_sktransformer_encode_need_to_concat_features_and_add_sample_dim():
    output_dim = 10
    n_vars = 3

    scaler = StandardScaler()
    scaler.fit(np.random.rand(5, n_vars * output_dim))
    pad_size = 5
    transformer = DummyTransformer(extra_embedded_dims=pad_size)
    sktransformer = SkTransformer(transformer, scaler, enforce_positive_outputs=False)
    x = [np.random.rand(output_dim) for i in range(n_vars)]

    assert sktransformer.encode(x).shape == (1, pad_size * 2 + n_vars * output_dim)


def test_sktransformer_encode_need_to_concat_features():
    output_dim = 10
    n_vars = 3

    scaler = StandardScaler()
    scaler.fit(np.random.rand(5, n_vars * output_dim))
    pad_size = 5
    transformer = DummyTransformer(extra_embedded_dims=pad_size)
    sktransformer = SkTransformer(transformer, scaler, enforce_positive_outputs=False)
    nt = 13
    x = [[np.random.rand(output_dim) for t in range(nt)] for i in range(n_vars)]
    assert sktransformer.encode(x).shape == (nt, pad_size * 2 + n_vars * output_dim)


def test_sktransformer_error_on_wrong_input_shape():
    output_dim = 10
    n_vars = 3

    scaler = StandardScaler()
    scaler.fit(np.random.rand(5, n_vars * output_dim))
    transformer = transformer = DummyTransformer(extra_embedded_dims=5)
    sktransformer = SkTransformer(transformer, scaler, enforce_positive_outputs=False)

    x = [np.random.rand(3, 3, 2, output_dim) for i in range(n_vars)]
    with pytest.raises(ValueError):
        sktransformer.predict(x)
