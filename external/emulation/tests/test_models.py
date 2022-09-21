import tensorflow as tf
import numpy as np
from emulation.models import ModelWithClassifier, transform_model
from emulation.zhao_carr import CLASS_NAMES


def test_ModelWithClassifier():
    inputs = {
        "a": tf.keras.Input(10, name="a"),
        "zero_tendency": tf.keras.Input(10, name="zero_tendency"),
    }
    outputs = {"out": tf.keras.layers.Lambda(lambda x: x, name="out")(inputs["a"])}
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    zero_tendency = tf.keras.layers.Dense(
        4, activation="softmax", name="gscond_classes"
    )(inputs["a"])
    classifier = tf.keras.Model([inputs["a"]], [zero_tendency])

    transformed_model = ModelWithClassifier(
        model, classifier, inputs_to_ignore=["singleton_vector"]
    )

    x = {"a": np.ones([100, 10])}

    # include singleton input in input vector to avoid batching related bugs
    x["singleton_vector"] = np.ones([1])

    out = transformed_model(x)
    assert out
    assert set(out) >= set(CLASS_NAMES)
    for v in out.values():
        assert isinstance(v, np.ndarray)


def test_ModelWithClassifier_no_classifier():
    inputs = {
        "a": tf.keras.Input(10, name="a"),
    }
    outputs = {"out": tf.keras.layers.Lambda(lambda x: x, name="out")(inputs["a"])}
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    transformed_model = ModelWithClassifier(model, classifier=None)

    x = {"a": np.ones([100, 10])}
    out = transformed_model(x)
    assert out
    for v in out.values():
        assert isinstance(v, np.ndarray)


def test_transform_model():
    x = {"a": np.ones([10, 100])}

    class Model:
        def __call__(self, x):
            return {"c": x["b"]}

    class Transform:
        def forward(self, x):
            a = x["a"]
            assert a.shape == (10, 100)
            x["b"] = a
            return x

        def backward(self, x):
            c = x["c"]
            assert c.shape == (10, 100)
            x["d"] = c
            return x

    model = Model()
    transform = Transform()
    f = transform_model(model, transform)
    out = f(x)
    assert set(out) == set("abcd")
