import datetime
import numpy as np
import tensorflow as tf
import xarray as xr
from runtime.emulator import (
    ScalarMLP,
    UVTQSimple,
    OnlineEmulator,
    OnlineEmulatorConfig,
    NormLayer,
    ScalarNormLayer,
    get_model,
)
import pytest

from runtime.loss import ScalarLoss


def test_OnlineEmulator_partial_fit(state):
    config = OnlineEmulatorConfig(
        batch_size=32, learning_rate=0.001, momentum=0.0, levels=63
    )

    emulator = OnlineEmulator(config)
    emulator.partial_fit(state, state)


def test_OnlineEmulator_partial_fit_logged(state):
    config = OnlineEmulatorConfig(
        batch_size=8, learning_rate=0.01, momentum=0.0, levels=63
    )
    time = datetime.datetime.now().isoformat()

    emulator = OnlineEmulator(config)
    writer = tf.summary.create_file_writer(f"logs_long/{time}")
    with writer.as_default():
        for i in range(10):
            emulator.partial_fit(state, state)


def test_OnlineEmulator_fails_when_accessing_nonexistant_var(state):
    config = OnlineEmulatorConfig(
        batch_size=32,
        learning_rate=0.001,
        momentum=0.0,
        extra_input_variables=["not a varialbe in any state 332r23r90e9d"],
        levels=63,
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
        levels=63,
    )

    emulator = OnlineEmulator(config)
    emulator.partial_fit(state, state)
    stateout = emulator.predict(state)
    assert isinstance(stateout, xr.Dataset)
    assert list(stateout["eastward_wind"].dims) == ["z", "y", "x"]


@pytest.mark.parametrize(
    "config",
    [
        OnlineEmulatorConfig(
            batch_size=32, learning_rate=0.001, momentum=0.0, levels=79
        ),
        OnlineEmulatorConfig(
            batch_size=32,
            learning_rate=0.001,
            momentum=0.0,
            target=ScalarLoss(0, 0),
            levels=79,
        ),
    ],
)
def test_OnlineEmulator_batch_fit(config):

    one = tf.zeros([79])

    u, v, t, q, dp = one, one, one, one, one

    dataset = tf.data.Dataset.from_tensors(((u, v, t, q, dp), (u, v, t, q, dp)))

    emulator = OnlineEmulator(config)
    emulator.batch_fit(dataset)


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


@pytest.mark.parametrize(
    "config, class_",
    [
        pytest.param(OnlineEmulatorConfig(), UVTQSimple, id="3d-out"),
        pytest.param(
            OnlineEmulatorConfig(target=ScalarLoss(0, 0)), ScalarMLP, id="scalar-mlp"
        ),
    ],
)
def test_get_model(config, class_):
    model = get_model(config)
    assert isinstance(model, class_)


@pytest.mark.parametrize("num_hidden_layers", [0, 1, 4])
def test_ScalarMLP(num_hidden_layers):
    ins = [tf.ones((1, 10), dtype=tf.float32)] * 5
    outs = [tf.zeros((1, 10), dtype=tf.float32)] * 5
    model = ScalarMLP(num_hidden_layers=num_hidden_layers)
    model.fit_scalers(ins, outs)
    out = model(ins)

    assert out.shape == (1, 1)

    # computing loss should not fail
    loss, _ = ScalarLoss(0, 0).loss(model, ins, outs)
    assert loss.shape == ()


def test_ScalarMLP_has_more_layers():
    shallow = ScalarMLP(num_hidden_layers=1)
    deep = ScalarMLP(num_hidden_layers=3)

    # build the models
    ins = [tf.ones((1, 10), dtype=tf.float32)] * 4
    outs = [tf.zeros((1, 10), dtype=tf.float32)] * 4
    for model in [shallow, deep]:
        model.fit_scalers(ins, outs)
        model(ins)

    assert len(deep.trainable_variables) > len(shallow.trainable_variables)


def test_top_level():
    dict_ = {"target": {"variable": 0, "level": 10}}
    config = OnlineEmulatorConfig.from_dict(dict_)
    assert ScalarLoss(0, 10) == config.target


@pytest.mark.parametrize("output_exists", [True, False])
def test_dump_load_OnlineEmulator(state, tmpdir, output_exists):
    if output_exists:
        path = str(tmpdir)
    else:
        path = str(tmpdir.join("model"))

    n = state["air_temperature"].sizes["z"]
    config = OnlineEmulatorConfig(levels=n)
    emulator = OnlineEmulator(config)
    emulator.partial_fit(state, state)
    emulator.dump(path)
    new_emulator = OnlineEmulator.load(path)

    # assert that the air_temperature output is unchanged
    field = "air_temperature"
    np.testing.assert_array_equal(
        new_emulator.predict(state)[field], emulator.predict(state)[field]
    )
