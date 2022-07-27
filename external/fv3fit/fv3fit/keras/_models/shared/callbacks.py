import dataclasses
import tensorflow as tf
import wandb

# Add third party callbacks here to make them available in training
THIRD_PARTY_CALLBACKS = {"WandbCallback": wandb.keras.WandbCallback}


def register_custom_callback(name: str):
    """
    Returns a decorator that will register the given custom keras callback
    to be usable in training configuration.

    example usage to add the callback CustomCallback to the registry:

    @register_custom_callback("custom_callback")
    class CustomCallback(tf.keras.callbacks):
        ...

    """

    def decorator(callback: tf.keras.callbacks.Callback) -> tf.keras.callbacks.Callback:
        THIRD_PARTY_CALLBACKS[name] = callback
        return callback

    return decorator


@dataclasses.dataclass
class CallbackConfig:
    """Configuration for adding callbacks to use during keras training.
    'name' should match the name of the callback class to use (case sensitive).
    This can be either a builtin keras callback or a third party callback.
    Available third party callbacks are registered in
    fv3fit.keras._models.shared.callbacks.THIRD_PARTY_CALLBACKS.
    """

    name: str
    kwargs: dict = dataclasses.field(default_factory=dict)

    @property
    def instance(self) -> tf.keras.callbacks.Callback:
        try:
            cls = getattr(tf.keras.callbacks, self.name)
            return cls(**self.kwargs)
        except AttributeError:
            try:
                return THIRD_PARTY_CALLBACKS[self.name](**self.kwargs)
            except KeyError:
                raise ValueError(
                    f"callback {self.name} is not in the Keras library nor the list "
                    f"of usable third party callbacks {list(THIRD_PARTY_CALLBACKS)}"
                )


@register_custom_callback("EpochModelCheckpoint")
class EpochModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    """ The built-in keras checkpoint can only take sample intervals as an arg
    for saving at fixed intervals. The arg 'period' for epochs is deprecated.
    This class replicates that functionality.
    """

    def __init__(
        self,
        filepath,
        period=1,
        monitor="val_loss",
        verbose=0,
        save_best_only=False,
        save_weights_only=False,
        mode="auto",
        options=None,
        **kwargs,
    ):
        super(EpochModelCheckpoint, self).__init__(
            filepath,
            monitor,
            verbose,
            save_best_only,
            save_weights_only,
            mode,
            "epoch",
            options,
        )
        self.epochs_since_last_save = 0
        self.period = period

    def on_epoch_end(self, epoch, logs=None):
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save % self.period == 0:
            self._save_model(epoch=epoch, batch=None, logs=logs)

    def on_train_batch_end(self, batch, logs=None):
        pass
