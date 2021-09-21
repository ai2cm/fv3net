import tensorflow as tf

from ..layers import MeanFeatureStdDenormLayer, IncrementStateLayer


class FieldOutput(tf.keras.layers.Layer):

    def __init__(self, sample_out, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.unscaled = tf.keras.layers.Dense(sample_out.shape[-1])
        self.denorm = MeanFeatureStdDenormLayer()
        self.denorm.fit(sample_out)

    def call(self, tensor):

        x = self.unscaled(tensor)
        return self.denorm(x)


class ResidualOutput(FieldOutput):

    def __init__(self, sample_out, dt_sec, *args, **kwargs):

        super().__init__(sample_out, *args, **kwargs)

        self.increment = IncrementStateLayer(dt_sec)

    def call(self, tensors):

        field_input, tendency = tensors
        return self.increment([field_input, tendency])
