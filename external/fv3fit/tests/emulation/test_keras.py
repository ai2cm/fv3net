import pytest
import numpy as np
import tensorflow as tf

from fv3fit.emulation.keras import (
    CustomLoss,
    NormalizedMSE,
    StandardLoss,
    save_model,
    score_model,
)


def _get_model(feature_dim, num_outputs):

    in_ = tf.keras.Input(feature_dim)
    out = tf.keras.layers.Lambda(lambda x: x + 10)(in_)

    if num_outputs == 1:
        outputs = out
    else:
        outputs = [out] * num_outputs

    model = tf.keras.Model(inputs=in_, outputs=outputs)

    return model


def test_model_save(tmp_path):
    """keras models with custom loss cannot be easily saved
    after fitting"""

    in_ = tf.keras.Input(shape=[10])
    out = tf.keras.layers.Dense(10)(in_)
    model = tf.keras.Model(inputs={"a": in_}, outputs={"b": out})
    one = tf.ones((5, 10))
    data = {"a": one, "b": one}
    model.compile(optimizer="adam", metrics=["mae"], loss={"a": tf.keras.losses.MSE})
    model.fit({"a": data["a"]}, {"b": data["b"]})

    output_before_save = model(data)["b"]

    save_model(model, str(tmp_path))
    loaded = tf.keras.models.load_model(str(tmp_path / "model.tf"))

    np.testing.assert_array_equal(output_before_save, loaded(data)["b"])


def test_model_score():
    in_ = tf.keras.Input(shape=[10], name="a")
    out = tf.keras.layers.Lambda(lambda x: x)(in_)
    model = tf.keras.Model(inputs={"a": in_}, outputs={"b": out})

    one = tf.ones((5, 10))
    data = {"a": one, "b": one}

    scores, profiles = score_model(model, data)
    assert scores["mse/b"] == 0
    assert scores["bias/b"] == 0


def test_model_score_no_outputs():

    in_ = tf.keras.Input(shape=[10])
    model = tf.keras.Model(inputs={"a": in_}, outputs=[])

    one = tf.ones((10, 5))
    data = {"a": one}

    with pytest.raises(ValueError):
        score_model(model, data)


def test_NormalizeMSE():
    sample = np.array([[25.0], [75.0]])
    target = np.array([[50.0], [50.0]])

    mse_func = NormalizedMSE("mean_std", sample)
    mse = mse_func(target, sample)
    np.testing.assert_approx_equal(mse, 1.0, 6)


def test_CustomLoss():

    config = CustomLoss(
        normalization="mean_std",
        loss_variables=["fieldA", "fieldB"],
        metric_variables=["fieldC"],
        weights=dict(fieldA=2.0),
    )

    tensor = tf.random.normal((100, 2))

    names = ["fieldA", "fieldB", "fieldC", "fieldD"]
    samples = [tensor] * 4
    m = dict(zip(names, samples))

    config.prepare(m)

    model = _get_model(2, 4)

    assert len(config._loss) == 2
    assert "fieldA" in config._loss
    assert "fieldB" in config._loss
    for k, v in config._loss.items():
        assert isinstance(v, NormalizedMSE)
    assert config._weights["fieldA"] == 2.0
    assert config._weights["fieldB"] == 1.0

    assert len(config._metrics) == 1
    assert "fieldC" in config._metrics
    assert isinstance(config._metrics["fieldC"], NormalizedMSE)

    config.compile(model)


@pytest.mark.parametrize(
    "kwargs",
    [{}, dict(loss="mse", metrics=["mae"], weights=[2.0, 1.0])],
    ids=["defaults", "specified"],
)
def test_StandardLoss(kwargs):

    config = StandardLoss(**kwargs)
    model = _get_model(10, 2)
    config.compile(model)


def test_StandardLoss_prepare_arbitrary_kwargs():

    config = StandardLoss()

    # doen't do anything but shouldn't fail
    config.prepare()
    config.prepare(random_kwarg=1)

    model = _get_model(10, 2)
    config.compile(model)
