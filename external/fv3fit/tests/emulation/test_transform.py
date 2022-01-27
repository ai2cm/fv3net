import pytest
import tensorflow as tf
from fv3fit.emulation.transforms import (
    LogTransform,
    ComposedTransformFactory,
    ComposedTransform,
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
    factory = ComposedTransformFactory(
        [TransformedVariableConfig("a", "transformed", mock_xform)]
    )
    transform = factory.build(x)
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
    transform = ComposedTransformFactory(
        [TransformedVariableConfig("a", "b", LogTransform())]
    )
    assert transform.backward_names({"b"}) == {"a"}
    assert transform.backward_names({"b", "random"}) == {"a", "random"}


def test_ComposedTransform_forward_backward_on_sequential_transforms():
    # some transforms could be mutually dependent

    class Rename:
        def __init__(self, in_name, out_name):
            self.in_name = in_name
            self.out_name = out_name

        def forward(self, x):
            return {self.out_name: x[self.in_name]}

        def backward(self, y):
            return {self.in_name: y[self.out_name]}

    transform = ComposedTransform([Rename("a", "b"), Rename("b", "c")])
    data = {"a": tf.ones((1,))}
    assert set(transform.forward(data)) == {"c"}
    assert set(transform.backward(transform.forward(data))) == set(data)
