import tensorflow as tf
from typing import Callable, Mapping, Tuple

TensorDict = Mapping[str, tf.Tensor]
LossFunction = Callable[[TensorDict, TensorDict], Tuple[tf.Tensor, TensorDict]]
