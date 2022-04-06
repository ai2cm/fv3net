import dataclasses
import tensorflow as tf
from typing import List

import wandb


THIRD_PARTY_CALLBACKS = {"WandbCallback": wandb.keras.WandbCallback}


@dataclasses.dataclass
class KerasCallbackConfig:
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


@dataclasses.dataclass
class CallbackConfig:
    callbacks: List[KerasCallbackConfig] = dataclasses.field(default_factory=list)
