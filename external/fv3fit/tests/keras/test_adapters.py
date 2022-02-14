from typing import Mapping
import numpy as np
import pytest
import tensorflow as tf
from fv3fit.keras.adapters import (
    get_inputs,
    ensure_dict_output,
    rename_dict_output,
    rename_dict_input,
    _ensure_list_input,
)


@pytest.mark.xfail
def test_get_inputs_already_mapping():

    in_ = {"a": tf.keras.Input(shape=[2])}
    out = tf.keras.layers.Lambda(lambda x: x)(in_)
    model = tf.keras.Model(inputs=in_, outputs=out)

    retrieved_inputs = get_inputs(model)

    assert isinstance(retrieved_inputs, Mapping)
    assert "a" in retrieved_inputs
    tf.debugging.assert_equal(retrieved_inputs["a"], in_["a"])


def test_get_inputs_no_input_raises():
    model = tf.keras.Model()
    with pytest.raises(ValueError):
        get_inputs(model)


def test_get_inputs_named_inputs():
    in_ = [tf.keras.Input(2, name="a")]
    out = tf.keras.layers.Lambda(lambda x: x)(in_)
    model = tf.keras.Model(inputs=in_, outputs=out)

    retrieved_inputs = get_inputs(model)

    assert isinstance(retrieved_inputs, Mapping)
    assert "a" in retrieved_inputs
    tf.debugging.assert_equal(retrieved_inputs["a"], in_[0])


def test_ensure_dict_output_multiple_out():
    i = tf.keras.Input(shape=[5])
    out_1 = tf.keras.layers.Dense(5, name="a")(i)
    out_2 = tf.keras.layers.Dense(5, name="b")(i)
    model = tf.keras.Model(inputs=[i], outputs=[out_1, out_2])

    model_dict_output = ensure_dict_output(model)

    i = tf.ones((4, 5))
    dict_out = model_dict_output(i)
    list_out = model(i)

    np.testing.assert_array_equal(dict_out["a"], list_out[0])
    np.testing.assert_array_equal(dict_out["b"], list_out[1])


def test_ensure_dict_output_single_out():
    i = tf.keras.Input(shape=[5])
    out_1 = tf.keras.layers.Dense(5, name="a")(i)
    model = tf.keras.Model(inputs=[i], outputs=[out_1])

    model_dict_output = ensure_dict_output(model)
    i = tf.ones((4, 5))
    dict_out = model_dict_output(i)
    output_tensor = model(i)

    np.testing.assert_array_equal(dict_out["a"], output_tensor)


def test_ensure_dict_output_already_dict_output():
    i = tf.keras.Input(shape=[5])
    out_1 = tf.keras.layers.Dense(5)(i)
    model = tf.keras.Model(inputs=[i], outputs={"a": out_1})

    model_dict_output = ensure_dict_output(model)
    i = tf.ones((4, 5))
    dict_out = model_dict_output(i)
    output_tensor = model(i)

    np.testing.assert_array_equal(dict_out["a"], output_tensor["a"])


def test_rename_dict_outputs():
    in_ = tf.keras.layers.Input(shape=(5,))
    a = tf.keras.layers.Dense(5)(in_)
    b = tf.keras.layers.Dense(5)(in_)
    out_ = {"a": a, "b": b}
    model = tf.keras.Model(inputs=in_, outputs=out_)

    renamed_model = rename_dict_output(model, {"a": "new_a"})
    one = tf.ones((1, 5))

    new_out = renamed_model(one)
    old_out = model(one)

    np.testing.assert_array_equal(new_out["new_a"], old_out["a"])
    np.testing.assert_array_equal(new_out["b"], old_out["b"])


def test_ensure_dict_output_has_correct_output_names():
    in_ = tf.keras.layers.Input(shape=(5,))
    a = tf.keras.layers.Dense(5, name="not_a")(in_)
    b = tf.keras.layers.Dense(5, name="not_b")(in_)
    out_ = {"a": a, "b": b}
    model = tf.keras.Model(inputs=in_, outputs=out_)
    fixed_model = ensure_dict_output(model)
    assert set(fixed_model.output_names) == set(out_)


def test_rename_dict_inputs():
    inputs = {
        "input_0": tf.keras.layers.Input(shape=(5,), name="input_0"),
        "input_1": tf.keras.layers.Input(shape=(5,), name="input_1"),
    }
    concat_inputs = tf.keras.layers.Concatenate()([layer for layer in inputs.values()])
    a = tf.keras.layers.Dense(5, name="a")(concat_inputs)
    model = tf.keras.Model(inputs=inputs, outputs={"a": a})

    renamed_model = rename_dict_input(model, {"input_0": "input_0_renamed"})
    one = tf.ones((1, 5))

    new_out = renamed_model({"input_0_renamed": one * 5, "input_1": one})
    old_out = model({"input_0": one * 5, "input_1": one})
    np.testing.assert_array_equal(new_out["a"], old_out["a"])


def test_ensure_list_input_preserves_output_dicts():
    inputs = {
        "input_0": tf.keras.layers.Input(shape=(5,), name="input_0"),
        "input_1": tf.keras.layers.Input(shape=(5,), name="input_1"),
    }
    concat_inputs = tf.keras.layers.Concatenate()([layer for layer in inputs.values()])
    outputs = {
        "output_0": tf.keras.layers.Dense(5, name="output_0")(concat_inputs),
        "output_1": tf.keras.layers.Dense(5, name="output_1")(concat_inputs),
    }
    dict_input_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    list_input_model = _ensure_list_input(dict_input_model)

    one = tf.ones((1, 5))

    output_from_dict_input = dict_input_model({"input_0": one, "input_1": one})
    output_from_list_input = list_input_model([one, one])

    assert set(output_from_dict_input) == set(output_from_list_input)
    for key in output_from_dict_input:
        np.testing.assert_array_equal(
            output_from_dict_input[key], output_from_list_input[key]
        )
