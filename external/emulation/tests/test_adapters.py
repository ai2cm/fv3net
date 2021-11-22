import numpy as np
import tensorflow as tf
from emulation._emulate.adapters import convert_to_dict_output


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
