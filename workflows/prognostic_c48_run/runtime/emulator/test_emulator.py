import argparse
import datetime
import numpy as np
from runtime.emulator.thermo import RelativeHumidityBasis
import tensorflow as tf
import xarray as xr
from runtime.emulator.emulator import (
    RHScalarMLP,
    ScalarMLP,
    UVTQSimple,
    OnlineEmulatorConfig,
    OnlineEmulator,
    NormLayer,
    ScalarNormLayer,
    UVTRHSimple,
    get_model,
    get_emulator,
)
from runtime.emulator.loss import MultiVariableLoss, RHLoss, ScalarLoss
from .utils import _get_argsin
import pytest
from hypothesis import given
from hypothesis.strategies import lists, integers


def test_OnlineEmulator_partial_fit(state):
    config = OnlineEmulatorConfig(
        batch_size=32, learning_rate=0.001, momentum=0.0, levels=63
    )

    emulator = get_emulator(config)
    emulator.partial_fit(state, state)


def test_OnlineEmulator_partial_fit_logged(state):
    config = OnlineEmulatorConfig(
        batch_size=8, learning_rate=0.01, momentum=0.0, levels=63
    )
    time = datetime.datetime.now().isoformat()

    emulator = get_emulator(config)
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

    emulator = get_emulator(config)
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

    emulator = get_emulator(config)
    emulator.partial_fit(state, state)
    stateout = emulator.predict(state)
    assert isinstance(stateout, xr.Dataset)
    assert list(stateout["eastward_wind"].dims) == ["z", "y", "x"]


@pytest.mark.parametrize("with_validation", [True, False])
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
        OnlineEmulatorConfig(target=RHLoss(50), levels=79,),
    ],
)
def test_OnlineEmulator_batch_fit(config, with_validation):
    x = _get_argsin(config.levels)
    dataset = tf.data.Dataset.from_tensors((x.args, x.args)).unbatch()

    emulator = get_emulator(config)

    if with_validation:
        emulator.batch_fit(dataset, validation_data=dataset)
    else:
        emulator.batch_fit(dataset)


def test_UVTQSimple():
    model = UVTQSimple(10, 10, 10, 10)
    shape = (3, 10)
    argsin = _get_argsin(levels=10, n=3)
    out = model(argsin)

    assert tuple(out.u.shape) == shape
    assert tuple(out.v.shape) == shape
    assert tuple(out.T.shape) == shape
    assert tuple(out.q.shape) == shape


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
        pytest.param(
            OnlineEmulatorConfig(relative_humidity=True), UVTRHSimple, id="rh-mlp"
        ),
    ],
)
def test_get_model(config, class_):
    model = get_model(config)
    assert isinstance(model, class_)


@pytest.mark.parametrize("num_hidden_layers", [0, 1, 4])
def test_ScalarMLP(num_hidden_layers):
    ins = _get_argsin(n=1, levels=10)
    outs = ins
    model = ScalarMLP(num_hidden_layers=num_hidden_layers)
    model.fit_scalers(ins, outs)
    out = model(ins)

    assert out.shape == (1, 1)

    # computing loss should not fail
    loss, _ = ScalarLoss(0, 0).loss(model(ins), outs)
    assert loss.shape == ()


def test_ScalarMLP_has_more_layers():
    shallow = ScalarMLP(num_hidden_layers=1)
    deep = ScalarMLP(num_hidden_layers=3)

    # build the models
    ins = _get_argsin(n=1, levels=10)
    for model in [shallow, deep]:
        model.fit_scalers(ins, ins)
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
    emulator = get_emulator(config)
    emulator.partial_fit(state, state)
    emulator.dump(path)
    new_emulator = OnlineEmulator.load(path)

    # assert that the air_temperature output is unchanged
    field = "air_temperature"
    np.testing.assert_array_equal(
        new_emulator.predict(state)[field], emulator.predict(state)[field]
    )


def test_RHScalarMLP():
    argsin = _get_argsin(n=10, levels=10)
    tf.random.set_seed(1)
    mlp = RHScalarMLP(var_number=3)
    mlp(argsin)
    mlp.fit_scalers(argsin, argsout=argsin)
    out = mlp(argsin)
    assert not np.isnan(out.numpy()).any()
    assert np.all(out.numpy() < 1.2)
    assert np.all(out.numpy() > 0.0)

    loss = RHLoss(mlp.var_level)
    val, _ = loss.loss(mlp(argsin), argsin)
    assert val.numpy() >= 0.0
    assert val.numpy() < 10.0


@pytest.mark.parametrize(
    "args,loss_cls",
    [
        (["--level", "50"], ScalarLoss),
        (["--level", "50", "--relative-humidity"], RHLoss),
        (["--multi-output"], MultiVariableLoss),
        (["--multi-output", "--relative-humidity"], MultiVariableLoss),
    ],
)
def test_OnlineEmulatorConfig_register_parser(args, loss_cls):
    parser = argparse.ArgumentParser()
    OnlineEmulatorConfig.register_parser(parser)
    args = parser.parse_args(args)
    config = OnlineEmulatorConfig.from_args(args)
    assert isinstance(config, OnlineEmulatorConfig)
    assert isinstance(config.target, loss_cls)

    if args.relative_humidity:
        assert config.relative_humidity
    else:
        assert not config.relative_humidity


@given(lists(integers(min_value=0)))
def test_OnlineEmulatorConfig_multi_output_levels(levels):

    str_levels = ",".join(str(s) for s in levels)

    parser = argparse.ArgumentParser()
    OnlineEmulatorConfig.register_parser(parser)
    args = parser.parse_args(["--multi-output", "--levels", str_levels])
    config = OnlineEmulatorConfig.from_args(args)

    assert config.target.levels == levels


def test_UVTRHSimple():
    n = 2
    x = _get_argsin(n)
    y = RelativeHumidityBasis(
        [x.u + 1.0, x.v + 1.0, x.T + 1.0, x.rh + 0.01, x.dp, x.dz]
    )
    model = UVTRHSimple(n, n, n, n)
    model.fit_scalers(x, y)
    out = model(x)
    assert (out.q.numpy() > 0).all()
