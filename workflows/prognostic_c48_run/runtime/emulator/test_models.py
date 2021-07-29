import numpy as np
import pytest
import tensorflow as tf
from runtime.emulator.loss import QVLoss, RHLoss
from runtime.emulator.models import (
    RHScalarMLP,
    ScalarMLP,
    UVTQSimple,
    UVTRHSimple,
    V1QCModel,
)
from runtime.emulator.thermo import RelativeHumidityBasis
from runtime.emulator.emulator import get_model, OnlineEmulatorConfig

from .utils import _get_argsin


@pytest.mark.parametrize("with_scalars", [True, False])
def test_V1QCModel(with_scalars):
    x = _get_argsin(3)

    if with_scalars:
        n_scalars = 2
        x.scalars = [tf.random.uniform([x.q.shape[0], n_scalars])]
    else:
        n_scalars = 0

    model = V1QCModel(x.u.shape[1])
    assert not model.scalers_fitted
    model.fit_scalers(x, x)
    assert model.scalers_fitted
    y = model(x)
    assert isinstance(y, RelativeHumidityBasis)


def test_UVTQSimple():
    model = UVTQSimple(10, 10, 10, 10)
    shape = (3, 10)
    argsin = _get_argsin(levels=10, n=3)
    model.fit_scalers(argsin, argsin)
    out = model(argsin)

    assert tuple(out.u.shape) == shape
    assert tuple(out.v.shape) == shape
    assert tuple(out.T.shape) == shape
    assert tuple(out.q.shape) == shape


def test_tf_dataset_behaves_as_expected_for_tuples():
    u = tf.ones((10, 1))
    d = tf.data.Dataset.from_tensor_slices(((u, u), (u, u)))
    (out, _), (_, _) = next(iter(d))
    assert (1,) == tuple(out.shape)


@pytest.mark.parametrize("num_hidden_layers", [0, 1, 4])
def test_ScalarMLP(num_hidden_layers):
    ins = _get_argsin(n=1, levels=10)
    outs = ins
    model = ScalarMLP(num_hidden_layers=num_hidden_layers)
    model.fit_scalers(ins, outs)
    out = model(ins)

    assert out.shape == (1, 1)

    # computing loss should not fail
    loss, _ = QVLoss(0).loss(model(ins), outs)
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


def test_RHScalarMLP():
    argsin = _get_argsin(n=10, levels=10)
    tf.random.set_seed(1)
    mlp = RHScalarMLP()
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


def test_UVTRHSimple():
    n = 2
    x = _get_argsin(n)
    y = RelativeHumidityBasis(x.u + 1.0, x.v + 1.0, x.T + 1.0, x.rh + 0.01, x.dp, x.dz)
    model = UVTRHSimple(n, n, n, n)
    model.fit_scalers(x, y)
    out = model(x)
    assert (out.q.numpy() > 0).all()


@pytest.mark.parametrize(
    "config, class_",
    [
        pytest.param(OnlineEmulatorConfig(), UVTQSimple, id="3d-out"),
        pytest.param(
            OnlineEmulatorConfig(target=QVLoss(0)), ScalarMLP, id="scalar-mlp"
        ),
        pytest.param(
            OnlineEmulatorConfig(relative_humidity=True), UVTRHSimple, id="rh-mlp"
        ),
    ],
)
def test_get_model(config, class_):
    model = get_model(config)
    assert isinstance(model, class_)
