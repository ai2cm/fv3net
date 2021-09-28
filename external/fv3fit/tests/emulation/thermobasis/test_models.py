from fv3fit.emulation.thermobasis.thermo import ThermoBasis
import numpy as np
import pytest
import tensorflow as tf
from fv3fit.emulation.layers import StableEncoder
from fv3fit.emulation.thermobasis.loss import QVLossSingleLevel, RHLossSingleLevel
from fv3fit.emulation.thermobasis.models import (
    RHScalarMLP,
    ScalarMLP,
    UVTQSimple,
    UVTRHSimple,
    V1QCModel,
    VectorModelAdapter,
)
from fv3fit.emulation.thermobasis.thermo import RelativeHumidityBasis
from fv3fit.emulation.thermobasis.emulator import (
    get_model,
    Config as OnlineEmulatorConfig,
)

from utils import _get_argsin


@pytest.mark.parametrize("with_scalars", [True, False])
def test_V1QCModel_outputs_a_relative_humidity_basis(with_scalars):
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


def test_UVTQSimple_outputs_the_same_shape():
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
def test_ScalarMLP_integration_with_loss(num_hidden_layers):
    ins = _get_argsin(n=1, levels=10)
    outs = ins
    model = ScalarMLP(num_hidden_layers=num_hidden_layers)
    model.fit_scalers(ins, outs)
    out = model(ins)

    assert out.shape == (1, 1)

    # computing loss should not fail (one simple integration)
    loss, _ = QVLossSingleLevel(0).loss(model(ins), outs)
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


def test_RHScalarMLP_integrations():
    """Tests that RH outputs and loss values are within reasonable ranges
    """
    argsin = _get_argsin(n=10, levels=10)
    tf.random.set_seed(1)
    mlp = RHScalarMLP()
    mlp(argsin)
    mlp.fit_scalers(argsin, argsout=argsin)
    out = mlp(argsin)
    assert not np.isnan(out.numpy()).any()
    assert np.all(out.numpy() < 1.2)
    assert np.all(out.numpy() > 0.0)

    loss = RHLossSingleLevel(mlp.var_level)
    val, _ = loss.loss(mlp(argsin), argsin)
    assert val.numpy() >= 0.0
    assert val.numpy() < 10.0


def test_UVTRHSimple_humidity_is_positive():
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
            OnlineEmulatorConfig(target=QVLossSingleLevel(0)),
            ScalarMLP,
            id="scalar-mlp",
        ),
        pytest.param(
            OnlineEmulatorConfig(relative_humidity=True), UVTRHSimple, id="rh-mlp"
        ),
    ],
)
def test_get_model(config, class_):
    model = get_model(config)
    assert isinstance(model, class_)


def _get_vector_model_adapter():
    class VectorModel(tf.keras.layers.Layer):
        def call(self, in_):
            prog, aux = in_
            self.add_loss(1.0)
            return prog

        def fit_scalers(self, x_in, x_next, aux):
            pass

    def expected_output(x):
        return x

    return VectorModelAdapter(VectorModel()), expected_output


def tensor_assert_almost_equal(x, y, rtol=None):
    return np.testing.assert_allclose(x.numpy(), y.numpy(), rtol=rtol)


def assert_relative_humidity_basis_almost_equal(
    x: ThermoBasis, y: ThermoBasis, rtol=1e-5
):
    tensor_assert_almost_equal(x.u, y.u, rtol=rtol)
    tensor_assert_almost_equal(x.rh, y.rh, rtol=rtol)
    tensor_assert_almost_equal(x.v, y.v, rtol=rtol)
    tensor_assert_almost_equal(x.T, y.T, rtol=rtol)
    tensor_assert_almost_equal(x.qc, y.qc, rtol=rtol)
    tensor_assert_almost_equal(x.dp, y.dp, rtol=rtol)
    tensor_assert_almost_equal(x.rho, y.rho, rtol=rtol)
    for x_scalar, y_scalar in zip(x.scalars, y.scalars):
        tensor_assert_almost_equal(x_scalar, y_scalar, rtol=rtol)
    assert len(x.scalars) == len(y.scalars)


def test_VectorModelAdapter_output_matches_expected():
    x = _get_argsin(levels=10, n=3)
    model, expected_output = _get_vector_model_adapter()
    expected = expected_output(x)
    out = model(x)
    assert_relative_humidity_basis_almost_equal(out, expected, rtol=1e-7)


def test_VectorModelAdapter_StableEncoder_integration():
    x = _get_argsin(levels=10, n=3)
    model = VectorModelAdapter(StableEncoder())
    model.fit_scalers(x, x)
    model(x)
