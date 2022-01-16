import pytest
import tensorflow as tf
from fv3fit.emulation.transforms import (
    LogTransform,
    PerVariableTransform,
    TransformedVariableConfig,
)


def _to_float(x: tf.Tensor) -> float:
    return x.numpy().item()


def _assert_scalar_approx(expected, actual):
    assert _to_float(expected) == pytest.approx(_to_float(actual))


def test_LogTransform_forward_backward():
    transform = LogTransform()
    x = tf.constant(0.001)
    x_approx = transform.backward(transform.forward(x))
    _assert_scalar_approx(x, x_approx)


def _get_per_variable_mocks():
    class MockTransform:
        def forward(self, x):
            return x + 1

        def backward(self, y):
            return y - 1

    x = {"a": tf.constant(1.0), "b": tf.constant(1.0)}
    mock_xform = MockTransform()
    transform = PerVariableTransform(
        [TransformedVariableConfig("a", "transformed", mock_xform)]
    )
    expected_forward = {
        "a": tf.constant(1.0),
        "b": tf.constant(1.0),
        "transformed": tf.constant(mock_xform.forward(x["a"])),
    }

    return transform, x, expected_forward


def test_per_variable_transform_forward():
    transform, x, expected_forward = _get_per_variable_mocks()
    y = transform.forward(x)
    assert set(y) == set(expected_forward)
    for key in y:
        _assert_scalar_approx(expected_forward[key], y[key])


def test_per_variable_transform_round_trip():
    transform, x, _ = _get_per_variable_mocks()
    y = transform.backward(transform.forward(x))
    assert set(y) >= set(x)
    for key in x:
        _assert_scalar_approx(x[key], y[key])


def test_per_variable_transform_backward_names():
    transform = PerVariableTransform(
        [TransformedVariableConfig("a", "b", LogTransform())]
    )
    assert transform.backward_names({"b"}) == {"a"}
    assert transform.backward_names({"b", "random"}) == {"a", "random"}
