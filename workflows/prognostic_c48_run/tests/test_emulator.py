import datetime
import numpy as np
import tensorflow as tf
import xarray as xr
from runtime.emulator import (
    UVTQSimple,
    OnlineEmulator,
    OnlineEmulatorConfig,
    NormLayer,
    ScalarNormLayer,
)
import pytest


def test_OnlineEmulator_partial_fit(state):
    config = OnlineEmulatorConfig(batch_size=32, learning_rate=0.001, momentum=0.0,)

    emulator = OnlineEmulator(config)
    emulator.partial_fit(state, state)


def test_OnlineEmulator_partial_fit_logged(state):
    config = OnlineEmulatorConfig(batch_size=8, learning_rate=0.01, momentum=0.0,)
    time = datetime.datetime.now().isoformat()

    emulator = OnlineEmulator(config)
    writer = tf.summary.create_file_writer(f"logs_long/{time}")
    with writer.as_default():
        for i in range(10):
            emulator.partial_fit(state, state)


def test_OnlineEmulator_predict_raises(state):
    config = OnlineEmulatorConfig(batch_size=32, learning_rate=0.001, momentum=0.0,)

    emulator = OnlineEmulator(config)
    with pytest.raises(ValueError):
        emulator.predict(state)


def test_OnlineEmulator_fails_when_accessing_nonexistant_var(state):
    config = OnlineEmulatorConfig(
        batch_size=32,
        learning_rate=0.001,
        momentum=0.0,
        extra_input_variables=["not a varialbe in any state 332r23r90e9d"],
    )

    emulator = OnlineEmulator(config)
    with pytest.raises(KeyError):
        emulator.partial_fit(state, state)


@pytest.mark.parametrize(
    "extra_inputs", [[], ["cos_zenith_angle", "surface_pressure"]],
)
def test_OnlineEmulator_fit_predict(state, extra_inputs):
    config = OnlineEmulatorConfig(
        batch_size=32,
        learning_rate=0.001,
        momentum=0.0,
        extra_input_variables=extra_inputs,
    )

    emulator = OnlineEmulator(config)
    emulator.partial_fit(state, state)
    stateout = emulator.predict(state)
    assert isinstance(stateout, xr.Dataset)
    assert list(stateout["eastward_wind"].dims) == ["z", "y", "x"]


def test_UVTQSimple():
    model = UVTQSimple(10, 10, 10, 10)
    shape = (3, 10)
    u = tf.ones(shape)
    v = tf.ones(shape)
    t = tf.ones(shape)
    q = tf.ones(shape)
    up, vp, tp, qp = model([u, v, t, q])

    for v in [up, vp, tp, qp]:
        assert tuple(v.shape) == shape


def test_NormLayer():
    u = tf.Variable([[0.0], [1.0]], dtype=tf.float32)
    layer = NormLayer()
    layer.fit(u)
    norm = layer(u)
    expected = np.array([[-1.0], [1.0]])
    np.testing.assert_allclose(expected, norm, rtol=1e-6)


def test_NormLayer_no_trainable_variables():
    u = tf.Variable([[0.0], [1.0]], dtype=tf.float32)
    layer = NormLayer()
    layer(u)

    assert [] == layer.trainable_variables


def test_NormLayer_gradient_works():
    u = tf.Variable([[0.0, 0.0], [1.0, 2.0]], dtype=tf.float32)
    layer = NormLayer()
    layer(u)

    with tf.GradientTape() as tape:
        y = layer(u)
    (g,) = tape.gradient(y, [u])
    expected = 1 / (layer.sigma + layer.epsilon)
    np.testing.assert_array_almost_equal(expected, g[0, :])


def test_tf_dataset_behaves_as_expected_for_tuples():
    u = tf.ones((10, 1))
    d = tf.data.Dataset.from_tensor_slices(((u, u), (u, u)))
    (out, _), (_, _) = next(iter(d))
    assert (1,) == tuple(out.shape)


def test_scalar_norm_layer():
    input = np.array([[1, 2], [-1, -2]], dtype=np.float32)
    expected = input * np.sqrt(10 / 4)

    norm = ScalarNormLayer()
    norm.fit(input)

    np.testing.assert_allclose(norm(input).numpy(), expected)
