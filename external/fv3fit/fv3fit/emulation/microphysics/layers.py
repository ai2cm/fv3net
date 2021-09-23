import tensorflow as tf

from ..layers import get_norm_class, get_denorm_class, IncrementStateLayer


class FieldInput(tf.keras.layers.Layer):
    def __init__(self, *args, sample_in=None, normalize=None, selection=None, **kwargs):
        super().__init__(*args, **kwargs)

        if normalize is not None:
            self.normalize = get_norm_class(normalize)(name=f"normalized_{self.name}")
            self.normalize.fit(sample_in)
        else:
            self.normalize = tf.keras.layers.Lambda(
                lambda x: x, name=f"passthru_{self.name}"
            )

        self.selection = selection

    def call(self, tensor: tf.Tensor):

        tensor = self.normalize(tensor)
        if self.selection is not None:
            tensor = tensor[..., self.selection]

        return tensor


class FieldOutput(tf.keras.layers.Layer):
    def __init__(self, sample_out, *args, normalize=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.unscaled = tf.keras.layers.Dense(
            sample_out.shape[-1], name=f"unscaled_{self.name}"
        )

        if normalize is not None:
            self.denorm = get_denorm_class(normalize)(name=f"denormalized_{self.name}")
            self.denorm.fit(sample_out)
        else:
            self.denorm = tf.keras.layers.Lambda(lambda x: x)

    def call(self, tensor):

        tensor = self.unscaled(tensor)
        return self.denorm(tensor)


class ResidualOutput(FieldOutput):
    def __init__(self, sample_out, dt_sec, *args, **kwargs):

        super().__init__(sample_out, *args, **kwargs)

        self.increment = IncrementStateLayer(dt_sec, name=f"increment_{self.name}")

    def call(self, tensors):

        field_input, network_output = tensors
        tendency = super().call(network_output)
        return self.increment([field_input, tendency])


class CombineInputs(tf.keras.layers.Layer):
    def __init__(self, combine_axis, *args, expand=False, **kwargs):
        super().__init__(*args, **kwargs)

        self.combine_axis = combine_axis
        self.expand = expand

    def call(self, inputs):

        if self.expand:
            inputs = [
                tf.expand_dims(tensor, axis=self.combine_axis) for tensor in inputs
            ]

        return tf.concat(inputs, axis=self.combine_axis)


class RNNBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        *args,
        channels=256,
        dense_width=256,
        dense_depth=1,
        combine_inputs=True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if combine_inputs:
            self.combine = CombineInputs(-1, expand=True)
        else:
            self.combine = tf.keras.layers.Lambda(
                lambda x: x, name=f"comb_passthru_{self.name}"
            )

        self.rnn = tf.keras.layers.SimpleRNN(
            channels, activation="relu", go_backwards=True
        )
        self.dense = MLPBlock(
            width=dense_width, depth=dense_depth, combine_inputs=False
        )

    def call(self, inputs):
        combined = self.combine(inputs)
        rnn_out = self.rnn(combined)

        output = self.dense(rnn_out)

        return output


class MLPBlock(tf.keras.layers.Layer):
    def __init__(self, *args, width=256, depth=2, combine_inputs=True, **kwargs):
        super().__init__(*args, **kwargs)
        if combine_inputs:
            self.combine = CombineInputs(-1, expand=False)
        else:
            self.combine = tf.keras.layers.Lambda(
                lambda x: x, name=f"comb_passthru_{self.name}"
            )

        self.dense = [tf.keras.layers.Dense(width) for i in range(depth)]

    def call(self, inputs):
        combined = self.combine(inputs)
        outputs = combined

        for i in range(len(self.dense)):
            outputs = self.dense[i](outputs)

        return outputs
