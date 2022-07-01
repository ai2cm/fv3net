import tensorflow as tf
import numpy as np
from emulation.models import TransformedModelWithClassifier, _predict


def test_TransformedModel_with_classifier(tmp_path):
    inputs = {"a": tf.keras.Input(10, name="a")}
    outputs = {"out": tf.keras.layers.Lambda(lambda x: x, name="out")(inputs["a"])}
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    class MockModel:
        inner_model = model

        def forward(self, x):
            return x

        def backward(self, x):
            return x

    def classifier(x):
        x = x["a"]
        n, z = x.shape
        nclasses = 4
        one_hot = np.zeros([n, z, nclasses])
        one_hot[..., 0] = 1.0
        return {"gscond_classes": tf.convert_to_tensor(one_hot)}

    model = MockModel()
    transformed_model = TransformedModelWithClassifier(model, classifier)

    x = {"a": np.ones([10, 100])}
    out = transformed_model(x)
    assert out
    for v in out.values():
        assert isinstance(v, np.ndarray)


def test__predict():
    x = {"a": np.ones([10, 100]), "b": np.ones([100])}

    def call(x):
        return {"y": tf.constant(x["a"])}

    out = _predict(call, x)
    assert set(out) == {"y"}
    np.testing.assert_array_equal(x["a"], out["y"])
