import pytest
import tensorflow as tf


def save_and_load_saved_model(model, path):
    tf.saved_model.save(model, path)
    return tf.saved_model.load(path)


def save_and_load_keras(model, path):
    tf.keras.models.save_model(model, path)
    return tf.keras.models.load_model(path)


@pytest.mark.parametrize(
    "save_and_load", [save_and_load_saved_model, save_and_load_keras]
)
def test_dump_load_keras_model_with_dict(tmpdir, save_and_load):
    """Test whether tensorflow ser/de-ser work for models returning dictionaries"""

    class DummyModel(tf.keras.Model):
        def __init__(self):
            super().__init__()

        def _random_method(self):
            pass

        def call(self, in_):
            out = {}
            out["b1"], out["b2"] = in_, in_
            return out

    in_ = tf.ones((1, 3))

    model = DummyModel()
    # this line is very important or tensorflow cannot trace the graph of the module
    model(in_)

    loaded = save_and_load(model, str(tmpdir))

    out = loaded(in_)
    assert set(out) == {"b1", "b2"}
    assert tf.is_tensor(out["b1"])
    assert tf.is_tensor(out["b2"])
    assert not hasattr(loaded, "_random_method")


def test_keras_functional_list_outputs():
    """A test ensuring how keras works"""
    # build model
    a = tf.keras.Input(name="a", shape=[5])
    b = tf.keras.layers.Dense(3, name="b")(a)
    c = tf.keras.layers.Dense(3, name="c")(a)
    model = tf.keras.models.Model(inputs=[a], outputs=[b, c])

    a = tf.ones((1, 5))
    b, c = model(a)
    assert tf.is_tensor(b)
    assert tf.is_tensor(c)
    assert model.output_names == ["b", "c"]


def test_keras_functional_dict_outputs():
    a = tf.keras.Input(name="a", shape=[5])
    b = tf.keras.layers.Dense(3)(a)
    c = tf.keras.layers.Dense(3)(a)
    model = tf.keras.models.Model(inputs=[a], outputs={"b": b, "c": c})

    a = tf.ones((1, 5))
    out = model(a)
    assert set(out) == {"b", "c"}
    assert tf.is_tensor(out["b"])
    assert tf.is_tensor(out["c"])


def test_keras_functional_dict_outputs_singleton():
    a = tf.keras.Input(name="a", shape=[5])
    # make sure that `name`` is ignored
    b = tf.keras.layers.Dense(3, name="not_b")(a)
    model = tf.keras.models.Model(inputs=[a], outputs={"b": b})

    a = tf.ones((1, 5))
    out = model(a)
    assert set(out) == {"b"}


@pytest.mark.parametrize("unused_input_variable", [True, False])
def test_train_keras_with_dict_output(unused_input_variable):
    a = tf.keras.Input(name="a", shape=[5])
    # make sure that `name`` is ignored
    b = tf.keras.layers.Dense(5, name="b")(a)
    model = tf.keras.models.Model(inputs={"a": a}, outputs={"b": b})
    model.compile(loss={"b": tf.keras.losses.MSE})

    one = tf.ones((1, 5))
    # need to split input variables and outputs into separate dicts
    in_ = {"a": one}
    if unused_input_variable:
        in_["b_unused"] = one

    model.fit(in_, {"b": one}, epochs=1)
