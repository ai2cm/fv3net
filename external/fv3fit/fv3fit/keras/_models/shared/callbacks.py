import dataclasses
import tensorflow as tf
import wandb


THIRD_PARTY_CALLBACKS = {"WandbCallback": wandb.keras.WandbCallback}


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
                return THIRD_PARTY_CALLBACKS[self.name](**self.kwargs)
            except KeyError:
                raise ValueError(
                    f"callback {self.name} is not in the Keras library nor the list "
                    f"of usable third party callbacks {list(THIRD_PARTY_CALLBACKS)}"
                )


class EpochModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    def __init__(
        self,
        filepath,
        interval=1,
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
        self.interval = interval

    def on_epoch_end(self, epoch, logs=None):
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save % self.interval == 0:
            self._save_model(epoch=epoch, batch=None, logs=logs)

    def on_train_batch_end(self, batch, logs=None):
        pass
