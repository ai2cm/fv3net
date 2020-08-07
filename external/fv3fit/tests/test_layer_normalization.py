from fv3fit.keras._models.normalizer import LayerStandardScaler
import numpy as np
import pytest
import tempfile
import tensorflow as tf


@pytest.fixture(params=["standard"])
def scaler(request):
    if request.param == "standard":
        return LayerStandardScaler()
    else:
        raise NotImplementedError()


@pytest.mark.parametrize("n_samples, n_features", [(10, 1), (10, 5)])
def test_normalize_layer_properties(scaler, n_samples, n_features):
    X = np.random.uniform(0, 10, size=[n_samples, n_features])
    scaler.fit(X)
    layer = scaler.normalize_layer
    assert isinstance(layer, tf.keras.layers.Layer)
    assert len(layer.trainable_weights) == 0


@pytest.mark.parametrize("n_samples, n_features", [(10, 1), (10, 5)])
def test_denormalize_layer_properties(scaler, n_samples, n_features):
    X = np.random.uniform(0, 10, size=[n_samples, n_features])
    scaler.fit(X)
    layer = scaler.denormalize_layer
    assert isinstance(layer, tf.keras.layers.Layer)
    assert len(layer.trainable_weights) == 0


@pytest.mark.parametrize("n_samples, n_features", [(10, 1), (10, 5)])
def test_standard_scaler_normalize_layer(n_samples, n_features):
    scaler = LayerStandardScaler()
    X = np.random.uniform(0, 10, size=[n_samples, n_features])
    scaler.fit(X)
    inputs = tf.keras.Input((n_features,))
    outputs = scaler.normalize_layer(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    result = model.predict(X)
    np.testing.assert_almost_equal(np.mean(result, axis=0), 0, decimal=6)
    np.testing.assert_almost_equal(np.std(result, axis=0), 1, decimal=6)


@pytest.mark.parametrize("n_samples, n_features", [(10, 1), (10, 5)])
def test_standard_scaler_normalize_layer_on_reloaded_scaler(n_samples, n_features):
    scaler = LayerStandardScaler()
    X = np.random.uniform(0, 10, size=[n_samples, n_features])
    scaler.fit(X)
    with tempfile.NamedTemporaryFile() as f_write:
        scaler.dump(f_write)
        with open(f_write.name, "rb") as f_read:
            scaler = scaler.load(f_read)
    inputs = tf.keras.Input((n_features,))
    outputs = scaler.normalize_layer(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    result = model.predict(X)
    np.testing.assert_almost_equal(np.mean(result, axis=0), 0, decimal=6)
    np.testing.assert_almost_equal(np.std(result, axis=0), 1, decimal=6)


@pytest.mark.parametrize("n_samples, n_features", [(10, 1), (10, 5)])
def test_scaler_normalize_then_denormalize_layer(scaler, n_samples, n_features):
    X = np.random.uniform(0, 10, size=[n_samples, n_features])
    scaler.fit(X)
    inputs = tf.keras.Input((n_features,))
    outputs = scaler.denormalize_layer(scaler.normalize_layer(inputs))
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    result = model.predict(X)
    np.testing.assert_almost_equal(result, X, decimal=6)


@pytest.mark.parametrize("n_samples, n_features", [(10, 1), (10, 5)])
def test_normalize_then_denormalize_on_reloaded_scaler(scaler, n_samples, n_features):
    scaler = LayerStandardScaler()
    X = np.random.uniform(0, 10, size=[n_samples, n_features])
    scaler.fit(X)
    result = scaler.normalize(X)
    with tempfile.NamedTemporaryFile() as f_write:
        scaler.dump(f_write)
        with open(f_write.name, "rb") as f_read:
            scaler = scaler.load(f_read)
    result = scaler.denormalize(result)
    np.testing.assert_almost_equal(result, X)


@pytest.mark.parametrize("n_samples, n_features", [(10, 1), (10, 5)])
def test_scaler_normalize_then_denormalize_layer_on_reloaded_scaler(
    scaler, n_samples, n_features
):
    X = np.random.uniform(0, 10, size=[n_samples, n_features])
    scaler.fit(X)
    with tempfile.NamedTemporaryFile() as f_write:
        scaler.dump(f_write)
        with open(f_write.name, "rb") as f_read:
            scaler = scaler.load(f_read)
    inputs = tf.keras.Input((n_features,))
    outputs = scaler.denormalize_layer(scaler.normalize_layer(inputs))
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    result = model.predict(X)
    np.testing.assert_almost_equal(result, X, decimal=6)
