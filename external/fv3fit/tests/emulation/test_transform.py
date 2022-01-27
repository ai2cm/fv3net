from unittest.mock import Mock
import numpy as np
import pytest
import tensorflow as tf
from fv3fit.emulation.transforms import (
    LogTransform,
    ComposedTransformFactory,
    ComposedTransform,
    TransformedVariableConfig,
)
from fv3fit.emulation.transforms.transforms import ConditionallyScaledTransform
from fv3fit.emulation.transforms.factories import ConditionallyScaled, fit_conditional
from fv3fit.emulation.zhao_carr_fields import Field


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


def test_fit_conditional():
    x = tf.convert_to_tensor([0.0, 1, 2])
    y = tf.convert_to_tensor([1.0, 2, 2])
    interp = fit_conditional(x, y, tf.reduce_mean, 2)
    # bins will be 0, 1, 2
    # mean is [1, 2]
    out = interp([-1, 0.0, 1, 3])
    expected = [1, 1, 2, 2]
    np.testing.assert_array_almost_equal(expected, out)


def test_fit_conditional_2d():
    # should work with multiple dimensions
    x = tf.convert_to_tensor([0.0, 1, 2])
    y = tf.convert_to_tensor([1.0, 2, 2])
    interp = fit_conditional(x, y, tf.reduce_mean, 2)

    in_ = tf.ones((10, 10))
    out = interp(in_)
    assert out.shape == in_.shape


def _get_mocked_transform(scale_value=2.0, min_scale=0.0):
    scale = Mock()
    scale.return_value = scale_value
    center = Mock()
    center.return_value = 0.0

    input_name = "x_in"
    output_name = "x_out"
    to = "difference"

    zero = tf.zeros((3, 4), dtype=tf.float32)

    expected = 1.0

    data = {
        input_name: zero,
        output_name: zero + expected * max(scale.return_value, min_scale),
    }

    transform = ConditionallyScaledTransform(
        input_name=input_name,
        output_name=output_name,
        to=to,
        on=input_name,
        scale=scale,
        center=center,
        min_scale=min_scale,
    )

    return scale, center, transform, data, expected, to


@pytest.mark.parametrize("min_scale", [0, 1, 2, 3])
def test_ConditionallyScaledTransform_forward(min_scale: float):
    scale, center, transform, data, expected, to = _get_mocked_transform(
        min_scale=min_scale
    )
    out = transform.forward(data)
    np.testing.assert_array_almost_equal(out[to], expected)
    scale.assert_called_once()
    center.assert_called_once()


@pytest.mark.parametrize("min_scale", [0, 4])
def test_ConditionallyScaledTransform_backward(min_scale: float):
    # need a scale-value other than one to find round-tripped errors
    scale, center, transform, data, expected, to = _get_mocked_transform(
        scale_value=2.0, min_scale=min_scale
    )
    out = transform.forward(data)
    round_tripped = transform.backward(out)
    assert set(round_tripped) == set(data) | set(out)
    for key in data:
        np.testing.assert_array_almost_equal(data[key], round_tripped[key])


def test_ConditionallyScaled_backward_names():
    in_name = "x_in"
    out_name = "x_out"
    on = "T"
    to = "diff"
    factory = ConditionallyScaled(
        field=Field(in_name, out_name), to=to, bins=10, condition_on=on
    )
    assert factory.backward_names({to}) == {in_name, on, out_name}


def test_ConditionallyScaled_backward_names_output_not_in_request():
    factory = ConditionallyScaled(
        field=Field("x", "y"), to="z", bins=10, condition_on="T"
    )
    assert factory.backward_names({"a", "b"}) == {"a", "b"}


def test_ConditionallyScaled_build():
    tf.random.set_seed(0)
    in_name = "x_in"
    out_name = "x_out"
    on = "T"
    to = "diff"
    factory = ConditionallyScaled(
        field=Field(in_name, out_name), to=to, bins=3, condition_on=on
    )
    shape = (10, 4)
    data = {
        in_name: tf.random.uniform(shape) * 5,
        out_name: tf.random.uniform(shape) * 3,
        on: tf.random.uniform(shape),
    }
    transform = factory.build(data)
    out = transform.forward(data)

    assert tf.reduce_mean(out[to]).numpy() == pytest.approx(0.0, abs=0.1)
    assert tf.reduce_mean(out[to] ** 2).numpy() == pytest.approx(1.0, abs=0.1)
