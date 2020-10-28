import tensorflow as tf
import tensorflow.keras.backend as K


class GCMCell(tf.keras.layers.AbstractRNNCell):
    def __init__(
        self,
        units: int,
        n_input: int,
        n_state: int,
        n_hidden_layers: int,
        # tendency_regularizer=None,
        kernel_regularizer=None,
        tendency_ratio: float = 0.1,
        dropout: float = 0.0,
        use_spectral_normalization: bool = True,
        **kwargs,
    ):
        self.units = units
        self.n_input = n_input
        self.n_state = n_state
        self.n_hidden_layers = n_hidden_layers
        self.activation = tf.keras.activations.relu
        self.tendency_ratio = tendency_ratio
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.kernel_regularizer = kernel_regularizer
        self.lock_to_inputs = False
        self.use_spectral_normalization = use_spectral_normalization
        self.us = {}
        """Set to True to set the GCMCell state exactly to the states you provide it.
        
        If False, the difference between consecutive inputs is still used as a forcing,
        but the GCMCell state is allowed to diverge from the input state.
        """
        # if tendency_regularizer is not None:
        #     self.tendency_regularizer = tendency_regularizer
        # else:
        #     self.tendency_regularizer = tf.keras.regularizers.l2(0.)
        super().__init__(**kwargs)

    @property
    def state_size(self):
        return self.n_state

    @property
    def output_size(self):
        return self.n_state

    def add_u(self, kernel, name):
        u = self.add_weight(
            shape=tuple([1, kernel.shape.as_list()[-1]]),
            initializer=tf.keras.initializers.RandomNormal(0, 1),
            name="u_" + name,
            trainable=False,
        )
        self.us[kernel.name] = u

    def spectral_normalize(self, kernel, training=None):
        u = self.us[kernel.name]
        v = self._l2normalize(K.dot(u, K.transpose(kernel)))
        new_u = self._l2normalize(K.dot(v, kernel))
        sigma = K.dot(v, kernel)
        sigma = K.dot(sigma, K.transpose(new_u))
        if training:
            u.assign(new_u)
        return kernel / sigma

    def _l2normalize(self, v, eps=1e-12):
        return v / (K.sum(v ** 2) ** 0.5 + eps)

    def build(self, input_shape):
        if len(input_shape) != 2:
            raise ValueError("expected two inputs, forcing and state")
        if len(input_shape[0]) != 2:
            raise ValueError(
                "expected to be used in stateful one-batch-at-a-time mode, "
                f"got input_shape {input_shape}"
            )
        n_input = input_shape[0][1]
        kernel = self.add_weight(
            shape=(n_input + self.n_state, self.units),
            initializer="glorot_uniform",
            name=f"dense_kernel_0",
        )
        if self.use_spectral_normalization:
            self.add_u(kernel, "0")
        bias = self.add_weight(
            shape=(self.units,), initializer="zeros", name="dense_bias_0"
        )
        self.dense_weights = [(kernel, bias)]
        for i in range(1, self.n_hidden_layers):
            kernel = self.add_weight(
                shape=(self.units, self.units),
                initializer="glorot_uniform",
                regularizer=self.kernel_regularizer,
                name=f"dense_kernel_{i}",
            )
            if self.use_spectral_normalization:
                self.add_u(kernel, str(i))
            bias = self.add_weight(
                shape=(self.units,), initializer="zeros", name=f"dense_bias_{i}"
            )
            self.dense_weights.append((kernel, bias))
        self.output_kernel = self.add_weight(
            shape=(self.units, self.n_state),
            initializer="glorot_uniform",
            name=f"output_kernel",
        )
        if self.use_spectral_normalization:
            self.add_u(self.output_kernel, "output")
        self.output_bias = self.add_weight(
            shape=(self.n_state,), initializer="zeros", name="output_bias"
        )
        self.built = True

    def call(self, inputs, states, training=None):
        gcm_state = states[0]
        input_forcing, input_state_delta = inputs
        # add tendencies from the host model
        gcm_state = tf.add(gcm_state, input_state_delta)
        h = K.concatenate([input_forcing, gcm_state], axis=-1)
        for kernel, bias in self.dense_weights:
            if self.use_spectral_normalization:
                kernel = self.spectral_normalize(kernel, training=training)
            h = self.activation(K.dot(h, kernel) + bias)
            h = self.dropout(h)
        if self.use_spectral_normalization:
            output_kernel = self.spectral_normalize(
                self.output_kernel, training=training
            )
        else:
            output_kernel = self.output_kernel
        tendency_output = K.dot(h, output_kernel) + self.output_bias
        # tendencies are on a much smaller scale than the variability of the data
        gcm_update = tf.multiply(
            tf.constant(self.tendency_ratio, dtype=tf.float32), tendency_output
        )
        gcm_output = gcm_state + gcm_update
        return gcm_output, [gcm_output]

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "n_input": self.n_input,
                "n_state": self.n_state,
                "n_hidden_layers": self.n_hidden_layers,
                "tendency_ratio": self.tendency_ratio,
                "lock_to_inputs": self.lock_to_inputs,
                "use_spectral_normalization": self.use_spectral_normalization,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        lock_to_inputs = config.pop("lock_to_inputs")
        obj = cls(**config)
        obj.lock_to_inputs = lock_to_inputs
        return obj
