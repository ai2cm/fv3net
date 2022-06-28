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
