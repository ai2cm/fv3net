import tensorflow as tf


class NormLayer(tf.keras.layers.Layer):
    def __init__(self, epsilon: float = 1e-7, name=None):
        super(NormLayer, self).__init__(name=name)
        self.epsilon = epsilon

    def build(self, in_shape):
        self.mean = self.add_weight(
            "mean", shape=[in_shape[-1]], dtype=tf.float32, trainable=False
        )
        self.sigma = self.add_weight(
            "sigma", shape=[in_shape[-1]], dtype=tf.float32, trainable=False
        )
        self.fitted = False

    def fit(self, tensor):
        self(tensor)
        self.mean.assign(tf.cast(tf.reduce_mean(tensor, axis=0), tf.float32))
        self.sigma.assign(
            tf.cast(
                tf.sqrt(tf.reduce_mean((tensor - self.mean) ** 2, axis=0)), tf.float32,
            )
        )

    def call(self, tensor):
        return (tensor - self.mean) / (self.sigma + self.epsilon)


class ScalarNormLayer(NormLayer):
    """UnNormalize a vector using a scalar mean and standard deviation


    """

    def __init__(self, name=None):
        super(ScalarNormLayer, self).__init__(name=name)

    def build(self, in_shape):
        self.mean = self.add_weight(
            "mean", shape=[in_shape[-1]], dtype=tf.float32, trainable=False
        )
        self.sigma = self.add_weight(
            "sigma", shape=[], dtype=tf.float32, trainable=False
        )

    def fit(self, tensor):
        self(tensor)
        self.mean.assign(tf.cast(tf.reduce_mean(tensor, axis=0), tf.float32))
        self.sigma.assign(
            tf.cast(tf.sqrt(tf.reduce_mean((tensor - self.mean) ** 2)), tf.float32,)
        )

    def call(self, tensor):
        return tensor * self.sigma + self.mean


class UnNormLayer(tf.keras.layers.Layer):
    def __init__(self, name=None):
        super(UnNormLayer, self).__init__(name=name)
        self.norm = NormLayer()

    @property
    def fitted(self):
        return self.norm.fitted

    def fit(self, x):
        self.norm.fit(x)

    def call(self, tensor):
        try:
            self.norm.mean
        except AttributeError:
            # need to initialize the weights
            self.norm(tensor)

        return tensor * self.norm.sigma + self.norm.mean
