import tensorflow as tf
import numpy as np
from emulation.models import ModelWithClassifier, _predict, transform_model
from emulation.zhao_carr import CLASS_NAMES


def test_TransformedModel_with_classifier(tmp_path):
    inputs = {
        "a": tf.keras.Input(10, name="a"),
        "zero_tendency": tf.keras.Input(10, name="zero_tendency"),
    }
    outputs = {"out": tf.keras.layers.Lambda(lambda x: x, name="out")(inputs["a"])}
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    def classifier(x):
        x = x["a"]
        n, z = x.shape
        nclasses = 4
        one_hot = np.zeros([n, z, nclasses])
        one_hot[..., 0] = 1.0
        return {"gscond_classes": tf.convert_to_tensor(one_hot)}

    transformed_model = ModelWithClassifier(model, classifier)

    x = {"a": np.ones([10, 100])}
    out = transformed_model(x)
    assert out
    assert set(out) >= set(CLASS_NAMES)
    for v in out.values():
        assert isinstance(v, np.ndarray)


def test__predict():
    x = {"a": np.ones([10, 100]), "b": np.ones([100])}

    def call(x):
        return {"y": tf.constant(x["a"])}

    out = _predict(call, x)
    assert set(out) == {"y"}
    np.testing.assert_array_equal(x["a"], out["y"])


def test_transform_model():
    x = {"a": np.ones([10, 100])}

    class Model:
        def __call__(self, x):
            return {"c": x["b"]}

    class Transform:
        def forward(self, x):
            a = x["a"]
            assert a.shape == (100, 10)
            x["b"] = a
            return x

        def backward(self, x):
            c = x["c"]
            assert c.shape == (100, 10)
            x["d"] = c
            return x

    model = Model()
    transform = Transform()
    f = transform_model(model, transform)
    out = f(x)
    assert set(out) == set("abcd")
