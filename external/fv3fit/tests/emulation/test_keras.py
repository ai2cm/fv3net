import pytest
import numpy as np
import tensorflow as tf

import fv3fit.emulation
from fv3fit.emulation.trainer import _ModelWrapper, train
from fv3fit.emulation.keras import (
    CustomLoss,
    NormalizedMSE,
    StandardLoss,
    save_model,
    score_model,
    get_jacobians,
    standardize_jacobians,
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


def test_train():
    model, data, loss = _get_model_data_loss()
    ds = tf.data.Dataset.from_tensors(data)
    train(model, ds, loss)
    assert not hasattr(train, "loss")


def test_train_wrong_shape_error():
    data = {"x": tf.ones((1, 10)), "y": tf.ones((1, 2, 3))}
    loss = CustomLoss(loss_variables=["y"], weights={"y": 1.0})

    def model(x):
        return {"y": tf.ones((2, 3))}

    ds = tf.data.Dataset.from_tensors(data)
    with pytest.raises(ValueError):
        train(model, ds, loss)


def _get_model_data_loss():
    in_ = tf.keras.Input(shape=[10], name="a")
    out = tf.keras.layers.Dense(10)(in_)
    model = tf.keras.Model(inputs={"a": in_}, outputs={"b": out})
    one = tf.ones((5, 10))
    data = {"a": one, "b": one}
    loss = CustomLoss(loss_variables=["b"], weights={"b": 1.0})
    loss.prepare(data)
    return model, data, loss


def test_checkpoint_callback(tmpdir):
    model, _, _ = _get_model_data_loss()
    trainer = _ModelWrapper(model)
    callback = fv3fit.emulation.ModelCheckpointCallback(
        filepath=str(tmpdir.join("{epoch:03d}.tf"))
    )
    callback.set_model(trainer)
    epoch = 0
    callback.on_epoch_end(epoch)
    tf.keras.models.load_model(callback.filepath.format(epoch=epoch))


def test_model_save(tmp_path):
    """keras models with custom loss cannot be easily saved
    after fitting"""

    model, data, _ = _get_model_data_loss()
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


def test_jacobians():
    sample = {
        "a": tf.random.normal((100, 5)),
        "b": tf.random.normal((100, 5)),
    }
    sample["field"] = sample["a"] + sample["b"]
    
    profiles = {
        name: tf.reduce_mean(sample[name], axis=0, keepdims=True)
        for name in ["a", "b"]    
    }

    inputs = {"a": tf.keras.Input(5), "b": tf.keras.Input(5)}
    out = tf.keras.layers.Lambda(lambda x: {"field": x["a"] + x["b"]})(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=out)

    jacobians = standardize_jacobians(
        get_jacobians(model, profiles),
        sample,
    )

    assert set(jacobians) == {"field"}
    assert set(jacobians["field"]) == {"a", "b"}
    for j in jacobians["field"].values():
        assert j.shape == (5, 5)
