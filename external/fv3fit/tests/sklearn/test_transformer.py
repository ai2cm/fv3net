from fv3fit.sklearn.transformer import SkTransformer
import numpy as np
import os
from sklearn.preprocessing import StandardScaler


class DummyTransformer:
    def __init__(self, latent_dim, output_dim, encoded_constant, decoded_constant):
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.encoded_constant = encoded_constant
        self.decoded_constant = decoded_constant

    def transform(self, X):
        return np.ones(self.latent_dim) + self.encoded_constant

    def inverse_transform(self, X):
        return np.ones(self.output_dim) + self.decoded_constant


def test_sktransformer_roundtrip():
    latent_dim = 5
    output_dim = 10
    n_vars = 3
    encoded_constant = 0
    decoded_constant = -2

    scaler = StandardScaler()
    scaler.fit(np.random.rand(5, n_vars * output_dim))
    transformer = transformer = DummyTransformer(
        latent_dim, n_vars * output_dim, encoded_constant, decoded_constant
    )
    sktransformer = SkTransformer(transformer, scaler, enforce_positive_outputs=False)

    x = [np.random.rand(output_dim) for i in range(n_vars)]
    target = np.ones(n_vars * output_dim) + decoded_constant
    outputs = sktransformer.predict(x)

    denormed_target = sktransformer.scaler.inverse_transform(target.reshape(1, -1))
    np.testing.assert_allclose(np.concatenate(outputs, axis=-1), denormed_target)


def test_sktransformer_enforce_positive():
    latent_dim = 5
    output_dim = 3
    n_vars = 2
    encoded_constant = 0
    decoded_constant = -5

    scaler = StandardScaler()
    scaler.fit(np.random.rand(5, n_vars * output_dim))
    transformer = transformer = DummyTransformer(
        latent_dim, n_vars * output_dim, encoded_constant, decoded_constant
    )
    sktransformer = SkTransformer(transformer, scaler, enforce_positive_outputs=False)
    sktransformer_enforce_positive = SkTransformer(
        transformer, scaler, enforce_positive_outputs=True
    )

    x = [np.arange(output_dim) * 10.0, np.arange(output_dim) * -10.0]
    outputs = np.concatenate(sktransformer.predict(x), axis=-1).reshape(-1)
    outputs_enforce_positive = np.concatenate(
        sktransformer_enforce_positive.predict(x), axis=-1
    ).reshape(-1)

    for o, o_positive in zip(outputs, outputs_enforce_positive):
        if o < 0.0:
            assert o_positive == 0.0
        else:
            assert o == o_positive


def test_sktransformer_dump_load(tmpdir):
    latent_dim = 5
    output_dim = 3
    n_vars = 2
    encoded_constant = 0
    decoded_constant = -5

    scaler = StandardScaler()
    scaler.fit(np.random.rand(5, n_vars * output_dim))
    transformer = transformer = DummyTransformer(
        latent_dim, n_vars * output_dim, encoded_constant, decoded_constant
    )
    sktransformer = SkTransformer(transformer, scaler, enforce_positive_outputs=True)
    x = [np.arange(output_dim) * 10.0, np.arange(output_dim) * -10.0]
    prediction = sktransformer.predict(x)

    path = os.path.join(str(tmpdir), "sktransformer")
    sktransformer.dump(path)

    loaded_sktransformer = SkTransformer.load(path)
    loaded_prediction = loaded_sktransformer.predict(x)

    for var, loaded_var in zip(prediction, loaded_prediction):
        np.testing.assert_allclose(var, loaded_var)
