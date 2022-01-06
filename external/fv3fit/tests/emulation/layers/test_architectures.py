import pytest
import numpy as np
import tensorflow as tf

from fv3fit.emulation.layers.architecture import (
    RNNBlock,
    _HiddenArchitecture,
    MLPBlock,
    HybridRNN,
    combine_inputs,
    NoWeightSharingSLP,
    RNNOutput,
    StandardOutput,
    ArchitectureConfig,
    _ARCHITECTURE_KEYS,
)


def _get_data(shape):

    num = int(np.prod(shape))
    return np.arange(num).reshape(shape).astype(np.float32)


def _get_tensor(shape):
    return tf.convert_to_tensor(_get_data(shape))


def test_MLPBlock():

    mlp = MLPBlock(width=256, depth=3)
    assert len(mlp.dense) == 3

    tensor = _get_tensor((20, 3))
    result = mlp(tensor)

    assert result.shape == (20, 256)


def test_MLPBlock_no_dense_layers():
    mlp = MLPBlock(width=256, depth=0)

    tensor = _get_tensor((20, 10))
    result = mlp(tensor)

    assert result.shape == (20, 10)


@pytest.mark.parametrize("depth,expected_shp", [(1, (20, 64)), (0, (20, 128))])
def test_HybridRNN(depth, expected_shp):

    rnn = HybridRNN(channels=128, dense_width=64, dense_depth=depth)

    tensor = _get_tensor((20, 10, 2))
    result = rnn(tensor)
    assert result.shape == expected_shp


def test_RNN():

    rnn = RNNBlock(channels=64, depth=2)
    tensor = _get_tensor((20, 10, 2))
    result = rnn(tensor)
    assert result.shape == (20, 10, 64)


def test_CombineInputs_no_expand():

    tensor = _get_tensor((20, 4))
    result = combine_inputs((tensor, tensor), -1, expand_axis=None)

    assert result.shape == (20, 8)
    np.testing.assert_array_equal(result[..., 4:8], tensor)


def test_CombineInputs_expand():

    tensor = _get_tensor((20, 4))
    result = combine_inputs((tensor, tensor, tensor), 2, expand_axis=2)

    assert result.shape == (20, 4, 3)
    np.testing.assert_array_equal(result[..., 2], tensor)


def test_no_weight_sharing_shape():
    tensor = tf.random.uniform([3, 4])
    model = NoWeightSharingSLP(5, 6)
    out = model(tensor)

    assert [3, 5] == list(out.shape)


def test_no_weight_sharing_num_weights():
    tensor = tf.random.uniform([3, 4])
    model = NoWeightSharingSLP(5, 6)
    model(tensor)

    num_weights_in_single_slp = 4 * 6 + 6 * 1 + 6 + 1
    num_weights_expected = 5 * (num_weights_in_single_slp)

    total = sum(np.prod(v.shape) for v in model.trainable_variables)

    assert num_weights_expected == total


@pytest.mark.parametrize("layer_cls", [MLPBlock, HybridRNN, RNNBlock])
def test_from_config(layer_cls):

    layer = layer_cls()
    config = layer.get_config()
    rebuilt = layer_cls.from_config(config)

    assert isinstance(rebuilt, layer_cls)


@pytest.mark.parametrize(
    "connector_cls, hidden_out",
    [(StandardOutput, _get_tensor((10, 64))), (RNNOutput, _get_tensor((10, 79, 64)))],
)
def test_OutputConnectors(connector_cls, hidden_out):

    feature_lens = {
        "field1": 79,
        "surface": 1,
    }

    connector = connector_cls(feature_lens)
    connected = connector(hidden_out)

    assert len(connected) == 2
    for key, output in connected.items():
        assert output.shape == (10, feature_lens[key])


def test_get_architecture_unrecognized():

    with pytest.raises(KeyError):
        ArchitectureConfig(name="not_an_arch")


@pytest.mark.parametrize("key", ["rnn-v1", "rnn", "dense", "linear"])
def test_ArchParams_bad_kwargs(key):
    with pytest.raises(TypeError):
        _HiddenArchitecture(key, dict(not_a_kwarg="hi"), {})


@pytest.mark.parametrize(
    "arch_key", _ARCHITECTURE_KEYS,
)
def test_ArchitectureConfig(arch_key):

    tensor = _get_tensor((10, 20))

    config = ArchitectureConfig(arch_key, {})
    output_features = {"out_field1": 20, "out_field2": 1}
    arch_layer = config.build(output_features)

    outputs = arch_layer([tensor, tensor])
    assert len(outputs) == 2
    for key, feature_len in output_features.items():
        assert key in outputs
        assert outputs[key].shape == (10, feature_len)


@pytest.mark.parametrize("rnn_key", ["rnn", "rnn-v1"])
def test_rnn_fails_with_inconsistent_vertical_dimensions(rnn_key):

    tensor1 = _get_tensor((10, 20))
    tensor2 = _get_tensor((10, 1))

    config = ArchitectureConfig(rnn_key)
    layer = config.build({})

    with pytest.raises(tf.errors.InvalidArgumentError):
        layer([tensor1, tensor2])
