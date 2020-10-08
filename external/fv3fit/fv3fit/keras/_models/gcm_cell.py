import tensorflow as tf
import tensorflow.keras.backend as K


class GCMCell(tf.keras.layers.AbstractRNNCell):
    def __init__(
        self, units: int, n_input: int, n_state: int, n_hidden_layers: int, **kwargs
    ):
        self.units = units
        self.n_input = n_input
        self.n_state = n_state
        self.n_hidden_layers = n_hidden_layers
        self.activation = tf.keras.activations.relu
        self.lock_to_inputs = False
        """Set to True to set the GCMCell state exactly to the states you provide it.
        
        If False, the difference between consecutive inputs is still used as a forcing,
        but the GCMCell state is allowed to diverge from the input state.
        """
        super().__init__(**kwargs)

    @property
    def state_size(self):
        return [self.n_state, self.n_state]

    @property
    def output_size(self):
        return self.n_state

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
        bias = self.add_weight(
            shape=(self.units,), initializer="zeros", name="dense_bias_0"
        )
        self.dense_weights = [(kernel, bias)]
        for i in range(1, self.n_hidden_layers):
            kernel = self.add_weight(
                shape=(self.units, self.units),
                initializer="glorot_uniform",
                name=f"dense_kernel_{i}",
            )
            bias = self.add_weight(
                shape=(self.units,), initializer="zeros", name=f"dense_bias_{i}"
            )
            self.dense_weights.append((kernel, bias))
        self.output_kernel = self.add_weight(
            shape=(self.units, self.n_state),
            initializer="glorot_uniform",
            name=f"output_kernel",
        )
        self.output_bias = self.add_weight(
            shape=(self.n_state,), initializer="zeros", name="output_bias"
        )
        self.built = True

    def call(self, inputs, states):
        gcm_state, last_input_state = states
        input_forcing, input_state = inputs
        # add tendencies from the host model
        input_state_diff = tf.subtract(input_state, last_input_state)
        gcm_state = tf.add(gcm_state, input_state_diff)
        h = K.concatenate([input_forcing, gcm_state])
        for kernel, bias in self.dense_weights:
            h = self.activation(K.dot(h, kernel) + bias)
        tendency_output = K.dot(h, self.output_kernel) + self.output_bias
        # tendencies are on a much smaller scale than the variability of the data
        gcm_update = tf.multiply(tf.constant(0.1, dtype=tf.float32), tendency_output)
        gcm_output = gcm_state + gcm_update
        if self.lock_to_inputs:
            # if this output is going to be applied to the Fortran model
            # with set_state, then the next time we calculate input_state_diff
            # we want input_state - (this gcm_output) to see what the
            # Fortran model has done.
            # This locks the GCMCell internal state to the input state,
            # regardless of whether you are truly updating
            # the inputs using this GCMCell
            reference_state = gcm_output
        else:
            # if the inputs are independent of this cell's output,
            # then input_state_diff is input_state - last_input_state
            # This allows the GCMCell internal state to evolve independently
            # of the inputs
            reference_state = input_state
        return gcm_output, [gcm_output, reference_state]

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "n_input": self.n_input,
                "n_state": self.n_state,
                "n_hidden_layers": self.n_hidden_layers,
                "lock_to_inputs": self.lock_to_inputs,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        lock_to_inputs = config.pop("lock_to_inputs")
        obj = cls(**config)
        obj.lock_to_inputs = lock_to_inputs
        return obj
