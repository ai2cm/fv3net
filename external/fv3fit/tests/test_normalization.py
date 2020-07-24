from fv3fit._shared.normalizer import StandardScaler
import numpy as np
import pytest
import tempfile
import tensorflow as tf


@pytest.fixture(params=["standard"])
def scaler(request):
    if request.param == "standard":
        return StandardScaler()
    else:
        raise NotImplementedError()


@pytest.mark.parametrize("n_samples, n_features", [(10, 1), (10, 5)])
def test_transform_layer_properties(scaler, n_samples, n_features):
    X = np.random.uniform(0, 10, size=[n_samples, n_features])
    scaler.fit(X)
    layer = scaler.transform_layer
    assert isinstance(layer, tf.keras.layers.Layer)
    assert len(layer.trainable_weights) == 0


@pytest.mark.parametrize("n_samples, n_features", [(10, 1), (10, 5)])
def test_inverse_transform_layer_properties(scaler, n_samples, n_features):
    X = np.random.uniform(0, 10, size=[n_samples, n_features])
    scaler.fit(X)
    layer = scaler.inverse_transform_layer
    assert isinstance(layer, tf.keras.layers.Layer)
    assert len(layer.trainable_weights) == 0


@pytest.mark.parametrize("n_samples, n_features", [(10, 1), (10, 5)])
def test_standard_scaler_transform(n_samples, n_features):
    scaler = StandardScaler()
    X = np.random.uniform(0, 10, size=[n_samples, n_features])
    scaler.fit(X)
    result = scaler.transform(X)
    np.testing.assert_almost_equal(np.mean(result, axis=0), 0)
    np.testing.assert_almost_equal(np.std(result, axis=0), 1)


@pytest.mark.parametrize("n_samples, n_features", [(10, 1), (10, 5)])
def test_standard_scaler_transform_layer(n_samples, n_features):
    scaler = StandardScaler()
    X = np.random.uniform(0, 10, size=[n_samples, n_features])
    scaler.fit(X)
    inputs = tf.keras.Input((n_features,))
    outputs = scaler.transform_layer(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    result = model.predict(X)
    np.testing.assert_almost_equal(np.mean(result, axis=0), 0, decimal=6)
    np.testing.assert_almost_equal(np.std(result, axis=0), 1, decimal=6)


@pytest.mark.parametrize("n_samples, n_features", [(10, 1), (10, 5)])
def test_standard_scaler_transform_layer_on_reloaded_scaler(n_samples, n_features):
    scaler = StandardScaler()
    X = np.random.uniform(0, 10, size=[n_samples, n_features])
    scaler.fit(X)
    with tempfile.NamedTemporaryFile() as f_write:
        scaler.dump(f_write)
        with open(f_write.name, "rb") as f_read:
            scaler = scaler.load(f_read)
    inputs = tf.keras.Input((n_features,))
    outputs = scaler.transform_layer(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    result = model.predict(X)
    np.testing.assert_almost_equal(np.mean(result, axis=0), 0, decimal=6)
    np.testing.assert_almost_equal(np.std(result, axis=0), 1, decimal=6)


@pytest.mark.parametrize("n_samples, n_features", [(10, 1), (10, 5)])
def test_transform_then_inverse_transform(scaler, n_samples, n_features):
    scaler = StandardScaler()
    X = np.random.uniform(0, 10, size=[n_samples, n_features])
    scaler.fit(X)
    result = scaler.inverse_transform(scaler.transform(X))
    np.testing.assert_almost_equal(result, X)


@pytest.mark.parametrize("n_samples, n_features", [(10, 1), (10, 5)])
def test_scaler_transform_then_inverse_transform_layer(scaler, n_samples, n_features):
    X = np.random.uniform(0, 10, size=[n_samples, n_features])
    scaler.fit(X)
    inputs = tf.keras.Input((n_features,))
    outputs = scaler.inverse_transform_layer(scaler.transform_layer(inputs))
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    result = model.predict(X)
    np.testing.assert_almost_equal(result, X, decimal=6)


@pytest.mark.parametrize("n_samples, n_features", [(10, 1), (10, 5)])
def test_transform_then_inverse_transform_on_reloaded_scaler(
    scaler, n_samples, n_features
):
    scaler = StandardScaler()
    X = np.random.uniform(0, 10, size=[n_samples, n_features])
    scaler.fit(X)
    result = scaler.transform(X)
    with tempfile.NamedTemporaryFile() as f_write:
        scaler.dump(f_write)
        with open(f_write.name, "rb") as f_read:
            scaler = scaler.load(f_read)
    result = scaler.inverse_transform(result)
    np.testing.assert_almost_equal(result, X)


@pytest.mark.parametrize("n_samples, n_features", [(10, 1), (10, 5)])
def test_scaler_transform_then_inverse_transform_layer_on_reloaded_scaler(
    scaler, n_samples, n_features
):
    X = np.random.uniform(0, 10, size=[n_samples, n_features])
    scaler.fit(X)
    with tempfile.NamedTemporaryFile() as f_write:
        scaler.dump(f_write)
        with open(f_write.name, "rb") as f_read:
            scaler = scaler.load(f_read)
    inputs = tf.keras.Input((n_features,))
    outputs = scaler.inverse_transform_layer(scaler.transform_layer(inputs))
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    result = model.predict(X)
    np.testing.assert_almost_equal(result, X, decimal=6)
