from fv3fit.keras._models.shared.pure_keras import PureKerasModel
import numpy as np
import tensorflow as tf

one = tf.ones((1, 5))
input_data = {"input_0": one, "input_1": one * 2.0}
output_data = {"output_0": one, "output_1": one * -2}


def pure_keras_model():

    input_0 = tf.keras.Input(name="input_0", shape=[5])
    input_1 = tf.keras.Input(name="input_1", shape=[5])
    input = tf.keras.layers.concatenate([input_0, input_1])

    dense = tf.keras.layers.Dense(5, name="dense_0")(input)

    output_0 = tf.keras.layers.Dense(5, name="output_0")(dense)
    output_1 = tf.keras.layers.Dense(5, name="output_1")(dense)

    losses = {"output_0": tf.keras.losses.MSE, "output_1": tf.keras.losses.MSE}
    # initial model has list inputs/outputs
    model = tf.keras.models.Model(
        inputs=[input_0, input_1], outputs=[output_0, output_1]
    )
    model.compile(loss=losses)

    model.fit(input_data, output_data, epochs=1)

    return PureKerasModel(
        input_variables=["input_0", "input_1"],
        output_variables=["output_0", "output_1"],
        model=model,
        output_metadata=None,
    )


def test_get_dict_compatible_model():
    pure_keras = pure_keras_model()
    input_positional = [input_data["input_0"], input_data["input_1"]]
    positional_prediction = pure_keras.model(input_positional)
    dict_prediction = pure_keras.get_dict_compatible_model()(input_data)
    assert len(dict_prediction) == len(positional_prediction)
    for i, output in enumerate(dict_prediction):
        assert np.allclose(dict_prediction[output], positional_prediction[i])
