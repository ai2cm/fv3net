import numpy as np
import tensorflow as tf
from fv3fit.keras.adapters import convert_to_dict_output, rename_dict_output


def test_convert_to_dict_output_multiple_out():
    i = tf.keras.Input(shape=[5])
    out_1 = tf.keras.layers.Dense(5, name="a")(i)
    out_2 = tf.keras.layers.Dense(5, name="b")(i)
    model = tf.keras.Model(inputs=[i], outputs=[out_1, out_2])

    model_dict_output = convert_to_dict_output(model)

    i = tf.ones((4, 5))
    dict_out = model_dict_output(i)
    list_out = model(i)

    np.testing.assert_array_equal(dict_out["a"], list_out[0])
    np.testing.assert_array_equal(dict_out["b"], list_out[1])


def test_convert_to_dict_output_single_out():
    i = tf.keras.Input(shape=[5])
    out_1 = tf.keras.layers.Dense(5, name="a")(i)
    model = tf.keras.Model(inputs=[i], outputs=[out_1])

    model_dict_output = convert_to_dict_output(model)
    i = tf.ones((4, 5))
    dict_out = model_dict_output(i)
    output_tensor = model(i)

    np.testing.assert_array_equal(dict_out["a"], output_tensor)


def test_convert_to_dict_output_already_dict_output():
    i = tf.keras.Input(shape=[5])
    out_1 = tf.keras.layers.Dense(5)(i)
    model = tf.keras.Model(inputs=[i], outputs={"a": out_1})

    model_dict_output = convert_to_dict_output(model)
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
