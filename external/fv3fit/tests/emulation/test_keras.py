import pytest
import numpy as np
import tensorflow as tf

from fv3fit.emulation.keras import save_model, score_model

def test_model_save(tmp_path):

    in_ = tf.keras.Input(10)
    out = tf.keras.layers.Lambda(lambda x: x + 10)(in_)
    model = tf.keras.Model(inputs=in_, outputs=out)

    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    save_model(model, str(tmp_path))

    loaded = tf.keras.models.load_model(str(tmp_path / "model.tf"))
    assert loaded.compiled_loss is None
    assert loaded.compiled_metrics is None


@pytest.fixture(params=[1, 2])
def model_and_target(request):
    in_ = tf.keras.Input(5)
    layer = tf.keras.layers.Lambda(lambda x: x + 10)(in_)
    tensor = tf.ones((10, 5)) + 10

    if request.param > 1:
        outputs = [layer for i in range(request.param)]
        target = [tensor for i in range(request.param)]
    else:
        outputs = layer
        target = tensor

    model = tf.keras.Model(inputs=in_, outputs=outputs)

    return model, target


def test_model_score_single(model_and_target):

    model, target = model_and_target

    in_tensor = tf.ones((10, 5))

    scores, profiles = score_model(model, in_tensor, target)

    _, score = scores.popitem()
    assert score == 0
