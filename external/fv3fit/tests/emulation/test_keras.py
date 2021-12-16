import pytest
import numpy as np
import tensorflow as tf

import fv3fit.emulation
from fv3fit.emulation.trainer import _ModelWrapper, train
from fv3fit.emulation.keras import (
    CustomLoss,
    NormalizedMSE,
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


def test_train():
    model, data, loss = _get_model_data_loss()
    ds = tf.data.Dataset.from_tensors(data)
    train(model, ds, loss)
    assert not hasattr(model, "loss")


def test_train_loss_integration():
    tf.random.set_seed(0)
    in_ = tf.keras.Input(2, name="x")
    out = tf.keras.layers.Dense(5, name="y1")(in_)
    out2 = tf.keras.layers.Dense(5, name="y2")(in_)
    model = tf.keras.Model(inputs=[in_], outputs={"y1": out, "y2": out2})
    loss = CustomLoss(loss_variables=["y1", "y2"], weights={"y1": 4.0, "y2": 1.0})

    batch_size = 10
    batch = {
        "x": tf.random.uniform((batch_size, 2)),
        "y1": tf.random.uniform((batch_size, 5)),
        "y2": tf.random.uniform((batch_size, 5)),
    }
    ds = tf.data.Dataset.from_tensor_slices(batch)
    loss.prepare(batch)
    history = train(model, ds.batch(2), loss=loss, epochs=3)

    # this magic number is regression data
    # update it if you expect this test to break
    assert history.history["loss"][-1] == pytest.approx(26.47243881225586)


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
    loss_fn = CustomLoss(
        normalization="mean_std",
        loss_variables=["fieldA", "fieldB"],
        metric_variables=["fieldC"],
        weights=dict(fieldA=2.0),
    )

    tensor = tf.random.normal((100, 2))

    names = ["fieldA", "fieldB", "fieldC", "fieldD"]
    samples = [tensor] * 4
    m = dict(zip(names, samples))
    loss_fn.prepare(m)

    # make a copy with some error
    compare = m.copy()

    loss, info = loss_fn(m, compare)
    assert set(info) == set(loss_fn.loss_variables) | set(loss_fn.metric_variables)
    assert loss.numpy() == pytest.approx(0.0)
