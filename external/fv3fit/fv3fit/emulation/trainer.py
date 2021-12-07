import tensorflow as tf


class Trainer(tf.keras.Model):
    """An object used for training another keras model
    
    This is used to separate training related serialization (e.g. loss
    functions, optimizers) from parameters needed for inference (e.g. weights
    and biases).
    """

    def __init__(self, model: tf.keras.Model):
        super().__init__()
        self.model = model

    def call(self, x):
        # non public any more...please don't use
        # needed for Trainer.fit to work
        return self.model(x)


class ModelCheckpointCallback(tf.keras.callbacks.Callback):
    """the built in one doesn't work with ``Trainer`` since it saves the full
    trainer object rather than the inner model"""

    def __init__(self, filepath: str):
        super().__init__()
        self.filepath = filepath

    def set_model(self, model: Trainer):
        self.model = model.model

    def on_epoch_end(self, epoch, logs=None):
        self.model.save(self.filepath.format(epoch=epoch), save_format="tf")
