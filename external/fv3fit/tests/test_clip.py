import numpy as np
import pytest
import tensorflow as tf

from fv3fit.keras._models.shared.clip import ClipConfig
from fv3fit._shared.config import SliceConfig


@pytest.mark.parametrize(
    "clip_config, expected",
    [
        pytest.param(
            ClipConfig({"var": {"z": SliceConfig(2, 6)}}),
            np.array([2, 3, 4, 5]),
            id="clip_1d",
        ),
        pytest.param(
            ClipConfig({"var": {"z": SliceConfig(8, None)}}),
            np.array([8, 9]),
            id="clip_1d_no_stop",
        ),
        pytest.param(
            ClipConfig({"var": {"z": SliceConfig(None, 2)}}),
            np.array([0, 1]),
            id="clip_1d_no_start",
        ),
        pytest.param(
            ClipConfig({"var_": {"z": SliceConfig(8, None)}}),
            np.arange(10),
            id="clip_var_not_in_config",
        ),
        pytest.param(
            ClipConfig({"var": {"z": SliceConfig(8, None), "y": SliceConfig(None, 2)}}),
            ValueError,
            id="too_many_clip_dims",
        ),
    ],
)
def test_ClipConfig_clip_along_last_dim_1d(clip_config, expected):
    layer = tf.keras.layers.Input(shape=(10,))

    if isinstance(expected, np.ndarray):
        clipped = clip_config.clip_along_last_dim(layer, "var")
        model = tf.keras.Model(inputs=layer, outputs=clipped)
        test_input = np.array([[i for i in range(10)]])
        predict = model.predict(test_input)
        assert np.array_equal(predict[0], expected)
    else:
        with pytest.raises(expected):
            clipped = clip_config.clip_along_last_dim(layer, "var")


def test_ClipConfig_clip_along_last_dim_2d():
    shape = (2, 10)
    layer = tf.keras.layers.Input(shape=shape)
    clip_config = ClipConfig({"var": {"z": SliceConfig(2, 6)}})
    clipped = clip_config.clip_along_last_dim(layer, "var")
    model = tf.keras.Model(inputs=layer, outputs=clipped)
    test_input = np.arange(20).reshape(1, *shape)
    predict = model.predict(test_input)
    assert np.array_equal(predict[0], np.array([[2, 3, 4, 5], [12, 13, 14, 15]]))


@pytest.mark.parametrize(
    "clip_config, expected",
    [
        pytest.param(
            ClipConfig({"var": {"z": SliceConfig(2, 5)}}),
            np.array([0, 0, 2, 3, 4, 0]),
            id="clip_both_ends",
        ),
        pytest.param(
            ClipConfig({"var": {"z": SliceConfig(2, None)}}),
            np.array([0, 0, 2, 3, 4, 5]),
            id="clip_start",
        ),
        pytest.param(
            ClipConfig({"var": {"z": SliceConfig(None, 5)}}),
            np.array([0, 1, 2, 3, 4, 0]),
            id="clip_end",
        ),
        pytest.param(ClipConfig(), np.array([0, 1, 2, 3, 4, 5]), id="no_clip"),
    ],
)
def test_ClipConfig_zero_mask_clipped_layer_1d(clip_config, expected):
    layer = tf.keras.layers.Input(6)
    masked = clip_config.zero_mask_clipped_layer(layer, "var")
    model = tf.keras.Model(inputs=layer, outputs=masked)
    test_input = np.arange(6)
    predict = model.predict(np.array([test_input]))
    assert np.array_equal(predict[0], expected)


def test_ClipConfig_zero_mask_clipped_layer_2d():
    clip_config = ClipConfig({"var": {"z": SliceConfig(2, 5)}})
    layer = tf.keras.layers.Input(shape=(2, 6))
    masked = clip_config.zero_mask_clipped_layer(layer, "var")
    model = tf.keras.Model(inputs=layer, outputs=masked)
    test_input = np.arange(12).reshape(2, 6)
    predict = model.predict(np.array([test_input]))
    expected = np.array([[0, 0, 2, 3, 4, 0], [0, 0, 8, 9, 10, 0]])
    assert np.array_equal(predict[0], expected)
