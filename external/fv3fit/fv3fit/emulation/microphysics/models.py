import tensorflow as tf

import fv3fit.emulation.layers as layers


def _dense_layers(inputs, width, depth):

    output = None
    for i in range(depth):
        dense_layer = tf.keras.layers.Dense(width, activation="relu")
        if i == 0:
            output = dense_layer(inputs)
        else:
            output = dense_layer(output)
    else:
        output = inputs

    return output


def get_rnn(inputs, channels=256, dense_width=256, dense_depth=1):

    expanded = [tf.expand_dims(tensor, axis=-1) for tensor in inputs]
    combined = tf.concat(expanded, axis=-1)
    recurrent = tf.keras.layers.SimpleRNN(channels, activation="relu", go_backwards=True, unroll=False)(combined)

    dense = _dense_layers(recurrent, dense_width, dense_depth)

    return dense


def get_dense(inputs, width=256, depth=2):
    combined = tf.concat(inputs, axis=-1)
    dense = _dense_layers(combined, width, depth)
    return dense


def get_norm_class(key):

    if key == "max_std":
        return layers.MaxFeatureStdNormLayer
    elif key == "mean_std":
        return layers.MeanFeatureStdNormLayer
    else:
        raise KeyError(f"Unrecognized normalization layer key provided: {key}")


def get_denorm_class(key):

    if key == "max_std":
        return layers.MaxFeatureStdDenormLayer
    elif key == "mean_std":
        return layers.MeanFeatureStdDenormLayer
    else:
        raise KeyError(f"Unrecognized de-normalization layer key provided: {key}")


def process_inputs(sample_in, names,  normalize=None, selection=None):

    inputs = []
    for name, sample in zip(names, sample_in):
        in_ = tf.keras.layers.Input(sample.shape[-1], name=name)

        if normalize is not None:
            norm_layer = get_norm_class(normalize)(name=f"normalize_{name}")
            norm_layer.fit(sample)
            in_ = norm_layer(in_)

        if selection is not None:
            in_ = in_[..., selection]

        inputs.append(in_)

    return inputs
