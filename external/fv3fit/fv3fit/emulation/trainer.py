from typing import Any, Sequence, Union
import tensorflow as tf
from fv3fit.emulation.keras import StandardLoss, CustomLoss


class _ModelWrapper(tf.keras.Model):
    def __init__(self, model: tf.keras.Model):
        super().__init__()
        self.model = model

    def call(self, x):
        return self.model(x)


def train(
    model: tf.keras.layers.Layer,
    dataset: Any,
    loss: Union[StandardLoss, CustomLoss],
    callbacks: Sequence[tf.keras.callbacks.Callback] = (),
    epochs: int = 1,
    validation_data: Any = None,
    validation_freq: Any = None,
    verbose: int = 0,
) -> Any:
    """Train a keras model with a specified loss function

    This is used to separate training related serialization (e.g. loss
    functions, optimizers) from parameters needed for inference (e.g. weights
    and biases).
    """
    wrapped_model = _ModelWrapper(model)
    loss.compile(wrapped_model)
    # explicitly handling all arguments makes it difficult to further expand
    # the surface area of this function
    # This minimizes our contact points to keras APIs...which are often
    # poorly documented
    return wrapped_model.fit(
        dataset,
        epochs=epochs,
        validation_data=validation_data,
        validation_freq=validation_freq,
        verbose=verbose,
        callbacks=callbacks,
    )


class ModelCheckpointCallback(tf.keras.callbacks.Callback):
    """the built in one doesn't work with ``Trainer`` since it saves the full
    trainer object rather than the inner model"""

    def __init__(self, filepath: str):
        super().__init__()
        self.filepath = filepath

    def set_model(self, model: _ModelWrapper):
        self.model = model.model

    def on_epoch_end(self, epoch, logs=None):
        self.model.save(self.filepath.format(epoch=epoch), save_format="tf")
