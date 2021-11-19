import numpy as np
import pytest
import tensorflow as tf

from fv3fit.keras._models.shared.clip import ClipConfig, zero_pad_output_feature_dim
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
            KeyError,
            id="clip_var_not_in_config",
        ),
        pytest.param(
            ClipConfig({"var": {"z": SliceConfig(8, None, 2)}}),
            NotImplementedError,
            id="step_slice",
        ),
        pytest.param(
            ClipConfig({"var": {"z": SliceConfig(8, None), "y": SliceConfig(None, 2)}}),
            ValueError,
            id="too_many_clip_dims",
        ),
    ],
)
def test_ClipConfig_clip_layer_along_feature_dim_1d(clip_config, expected):
    layer = tf.keras.layers.Input(shape=(10,))

    if isinstance(expected, np.ndarray):
        clipped = clip_config.clip_layer_along_feature_dim(layer, "var")
        model = tf.keras.Model(inputs=layer, outputs=clipped)
        test_input = np.array([[i for i in range(10)]])
        predict = model.predict(test_input)
        assert np.array_equal(predict[0], expected)
    else:
        with pytest.raises(expected):
            clipped = clip_config.clip_layer_along_feature_dim(layer, "var")


def test_ClipConfig_clip_layer_along_feature_dim_2d():
    shape = (2, 10)
    layer = tf.keras.layers.Input(shape=shape)
    clip_config = ClipConfig({"var": {"z": SliceConfig(2, 6)}})
    clipped = clip_config.clip_layer_along_feature_dim(layer, "var")
    model = tf.keras.Model(inputs=layer, outputs=clipped)
    test_input = np.arange(20).reshape(1, *shape)
    predict = model.predict(test_input)
    assert np.array_equal(predict[0], np.array([[2, 3, 4, 5], [12, 13, 14, 15]]))


@pytest.mark.parametrize(
    "original_size, clip_config, expected",
    [
        (10, ClipConfig({"var": {"z": SliceConfig(2, 6)}}), (2, 4)),
        (10, ClipConfig({"var": {"z": SliceConfig(None, 6)}}), (0, 4)),
        (10, ClipConfig({"var": {"z": SliceConfig(6, None)}}), (6, 0)),
        (10, ClipConfig({"var": {"z": SliceConfig(None, None)}}), (0, 0)),
    ],
)
def test_ClipConfig_removed_sizes(original_size, clip_config, expected):
    n_start_removed, n_end_removed = clip_config.removed_sizes(original_size, "var")
    assert (n_start_removed, n_end_removed) == expected


@pytest.mark.parametrize(
    "n_pad_start, n_pad_end, expected",
    [
        pytest.param(0, 0, np.array([1.0, 2.0]), id="no_padding_1d"),
        pytest.param(3, 0, np.array([0, 0, 0, 1.0, 2.0]), id="pad_start_1d"),
        pytest.param(0, 3, np.array([1.0, 2.0, 0, 0, 0]), id="pad_end_1d"),
        pytest.param(3, 2, np.array([0, 0, 0, 1.0, 2.0, 0.0, 0.0]), id="pad_both_1d"),
    ],
)
def test_zero_pad_output_feature_dim_1d(n_pad_start, n_pad_end, expected):
    clipped_layer = tf.keras.layers.Input(shape=2)
    filled = zero_pad_output_feature_dim(clipped_layer, n_pad_start, n_pad_end)
    model = tf.keras.Model(inputs=clipped_layer, outputs=filled)
    test_input = np.array([[1.0, 2.0]])
    predict = model.predict(test_input)
    assert np.array_equal(predict[0], expected)


@pytest.mark.parametrize(
    "clipped_output_shape, n_pad_start, n_pad_end",
    [
        pytest.param((2, 5), 1, 2, id="pad_both_2d"),
        pytest.param((2, 5), 0, 0, id="no_padding_2d"),
        pytest.param((2, 3, 5), 1, 2, id="pad_both_3d"),
        pytest.param((2, 3, 5), 0, 0, id="no_padding_3d"),
    ],
)
def test_zero_pad_output_feature_dim_higher_dims(
    clipped_output_shape, n_pad_start, n_pad_end
):
    layer = tf.keras.layers.Input(shape=clipped_output_shape)
    filled = zero_pad_output_feature_dim(layer, n_pad_start, n_pad_end)
    non_feature_dim_shape = layer.shape[1:-1]
    final_feature_dim_size = layer.shape[-1] + n_pad_start + n_pad_end
    assert filled.shape[1:] == (*non_feature_dim_shape, final_feature_dim_size)
