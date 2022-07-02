from fv3fit.emulation.types import LossFunction
from typing import Any, Sequence, Optional
import tensorflow as tf


class _ModelWrapper(tf.keras.Model):
    def __init__(self, model: tf.keras.Model):
        super().__init__()
        self.model = model

    def call(self, x):
        return self.model(x)


class _LayerTrainer(tf.keras.Model):
    """A class for training multiple output layers

    This uses the more explicit `add_loss` API for clarity
    """

    def __init__(self, model: tf.keras.Model, loss: LossFunction):
        super().__init__()
        self.model = model
        self._loss = loss

    def call(self, x):
        y = self.model(x)
        loss, metrics = self._loss(x, y)
        self.add_loss(loss)
        for var in metrics:
            self.add_metric(metrics[var], name=var)


def train(
    model: tf.keras.layers.Layer,
    dataset: tf.data.Dataset,
    loss: LossFunction,
    optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
    callbacks: Sequence[tf.keras.callbacks.Callback] = (),
    epochs: int = 1,
    validation_data: Optional[tf.data.Dataset] = None,
    validation_freq: Any = None,
    verbose: int = 0,
) -> Any:
    """Train a keras model with a specified loss function from dictionary data

    This works-around several awkward aspects of the keras APIs:
    - ``model.compile`` adds attributes to models that may not be
        serializeable
    - need to split the dictionary data into separate input/output dicts

    This function more cleanly separates the training related concerns and data
    (e.g. loss functions, optimizers) from parameters needed for inference (e.g.
    weights and biases).

    Args:
        model: is the keras layer to train
        dataset: a tensorflow dataset iterating over data dictionaries of type
            ``Mapping[str, tf.Tensor]``. This dictionary includes both inputs
            and target variables.
        validation_data: same as ``dataset`` but used for computing validation
            scores
        loss: a loss function ...loss_fn(truth,prediction) returns a scalar and
            dictionary of scalar metrics when truth, prediction are dicts of tensors
        optimizer: the optimizer. defaults to tf.keras.layers.Adam.

    Returns:
        The keras training history

    Note:
        all other arguments have the same interpretation as ``tf.keras.Model.fit

    """
    wrapped_model = _LayerTrainer(model, loss)
    train_set = next(iter(dataset))
    outputs = model(train_set)
    for name in outputs:
        if name in train_set:
            shape_in_data = tuple(train_set[name].shape)
            shape_in_output = tuple(outputs[name].shape)
            if shape_in_data != shape_in_output:
                raise ValueError(
                    f"{model} produced unexpected shape for {name}. "
                    f"Expected {shape_in_data} got {shape_in_output}."
                )

    wrapped_model.compile(optimizer=optimizer or tf.keras.optimizers.Adam())
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
