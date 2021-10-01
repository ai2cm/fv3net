import tensorflow as tf


class RNNBlock(tf.keras.layers.Layer):
    """
    RNN connected to an optional MLP for prediction

    Combines multiple tensors along a new trailing dimension and
    operates in the reverse on the feature dimension, i.e., from
    the top of the atmosphere to the bottom for microphysics data.
    """

    def __init__(
        self,
        *args,
        channels: int = 256,
        dense_width: int = 256,
        dense_depth: int = 1,
        activation: str = "relu",
        **kwargs,
    ):
        """
        Args:
            channels: width of RNN layer
            dense_width: width of MLP layer, only used if dense_depth > 0
            dense_depth: number of MLPlayers connected to RNN output, set to
                0 to disable
            activation: activation function to use for RNN and MLP
        """
        super().__init__(*args, **kwargs)

        self.rnn = tf.keras.layers.SimpleRNN(
            channels, activation=activation, go_backwards=True
        )
        self.dense = MLPBlock(width=dense_width, depth=dense_depth)

    def call(self, input):

        rnn_out = self.rnn(input)

        output = self.dense(rnn_out)

        return output


class MLPBlock(tf.keras.layers.Layer):
    """MLP layer for basic NN predictions"""

    def __init__(
        self,
        *args,
        width: int = 256,
        depth: int = 2,
        activation: str = "relu",
        **kwargs,
    ):
        """
        Args:
            width: width of MLP layer, only used if dense_depth > 0
            depth: number of MLPlayers, set to 0 to disable
            activation: activation function to use for RNN and MLP
        """
        super().__init__(*args, **kwargs)

        self.dense = [
            tf.keras.layers.Dense(width, activation=activation) for i in range(depth)
        ]

    def call(self, input):

        outputs = input

        for i in range(len(self.dense)):
            outputs = self.dense[i](outputs)

        return outputs