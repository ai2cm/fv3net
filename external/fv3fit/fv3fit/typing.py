from typing import Dict, Mapping, Protocol
import tensorflow as tf

Batch = Mapping[str, tf.Tensor]


# https://stackoverflow.com/questions/54668000/
class Dataclass(Protocol):
    __dataclass_fields__: Dict
