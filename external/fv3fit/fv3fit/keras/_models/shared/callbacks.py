import dataclasses
import tensorflow as tf
import wandb

# Add third party callbacks here to make them available in training
ADDITIONAL_CALLBACKS = {"WandbCallback": wandb.keras.WandbCallback}


def register_custom_callback(name: str):
    """
    Returns a decorator that will register the given training function
    to be usable in training configuration.
    """

    def decorator(callback: tf.keras.callbacks.Callback) -> tf.keras.callbacks.Callback:
        ADDITIONAL_CALLBACKS[name] = callback
        return callback

    return decorator


@dataclasses.dataclass
class CallbackConfig:
    name: str
    kwargs: dict = dataclasses.field(default_factory=dict)

    @property
    def instance(self) -> tf.keras.callbacks.Callback:
        try:
            cls = getattr(tf.keras.callbacks, self.name)
            return cls(**self.kwargs)
        except AttributeError:
            try:
                return ADDITIONAL_CALLBACKS[self.name](**self.kwargs)
            except KeyError:
                raise ValueError(
                    f"callback {self.name} is not in the Keras library nor the list "
                    f"of usable third party callbacks {list(ADDITIONAL_CALLBACKS)}"
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
