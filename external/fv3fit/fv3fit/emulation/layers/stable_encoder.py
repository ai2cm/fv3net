import tensorflow as tf
from fv3fit.emulation.layers import (
    MeanFeatureStdNormLayer,
    MeanFeatureStdDenormLayer,
    StandardNormLayer,
)


def mlp(num_hidden, num_layers, num_output):
    return tf.keras.Sequential(
        [tf.keras.layers.Dense(num_hidden, activation="relu")] * (num_layers - 1)
        + [tf.keras.layers.Dense(num_output, activation=None)]
    )


class StableDynamics(tf.keras.layers.Layer):
    def __init__(self, n: int):
        super(StableDynamics, self).__init__()
        self.phase = self.add_weight(
            "phase", shape=[n], dtype=tf.float32, trainable=True
        )
        self.growth_rate = self.add_weight(
            "growth_rate", shape=[n], dtype=tf.float32, trainable=True
        )

    def call(self, z):
        scale = -tf.nn.relu(self.growth_rate)
        exponent = tf.complex(scale, self.phase)
        return tf.exp(exponent) * z


class StableEncoder(tf.keras.layers.Layer):
    """A next-time-step predictor enforcing stable dynamics

    Stability is enforced by assuming that the dynamics are a described by
    damped oscillators in complex-valued latent space.

    Let x be the prognostic variables, and y the auxiliary variables.
    
    Then the next timestep prediction is given by ::

        x_next = g^{-1}(exp(i * phase - growth) g(x, y), y)


    """

    def __init__(self):
        super(StableEncoder, self).__init__()
        self.num_hidden = 128
        self.num_layers = 2

        self.prog_norm = MeanFeatureStdNormLayer()
        self.prog_denorm = MeanFeatureStdDenormLayer()
        self.aux_norm = StandardNormLayer()

    def build(self, input_shape):
        prog_shape, _ = input_shape
        num_prognostic = prog_shape[-1]

        # encoder
        self.encoder_real = mlp(self.num_hidden, self.num_layers, self.num_hidden)
        self.encoder_complex = mlp(self.num_hidden, self.num_layers, self.num_hidden)

        # dynamics
        self.dynamics = StableDynamics(self.num_hidden)

        # decoder
        self.decoder = mlp(self.num_hidden, self.num_layers, num_prognostic)

    def _decode(self, z, auxiliary):
        stacked = tf.concat([tf.math.real(z), tf.math.imag(z), auxiliary], axis=-1)
        return self.decoder(stacked)

    def fit_scalers(self, x_now, x_next, aux_now):
        self.prog_denorm.fit(x_now)
        self.prog_norm.fit(x_next)
        self.aux_norm.fit(aux_now)
        # TODO add some heuristic for estimating the time-scales
        # to initialize the StableDynamics better

    def call(self, in_):

        prognostic, auxiliary = in_

        prog_norm = self.prog_norm(prognostic)
        aux_norm = self.aux_norm(auxiliary)

        stacked = tf.concat([prog_norm, aux_norm], axis=-1)
        z = tf.complex(self.encoder_real(stacked), self.encoder_complex(stacked))
        z_next = self.dynamics(z)
        prognostic_decoded = self._decode(z, aux_norm)
        y_prediction = self._decode(z_next, aux_norm)

        # add auto-encoding loss
        self.add_loss(tf.reduce_mean((prognostic_decoded - prognostic) ** 2))

        return self.prog_denorm(y_prediction)
