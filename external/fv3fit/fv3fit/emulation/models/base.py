import tensorflow as tf
from typing import Mapping, Protocol, List


class Model(Protocol):
    @property
    def name(self) -> str:
        pass

    @property
    def output_variables(self) -> List[str]:
        pass

    @property
    def input_variables(self) -> List[str]:
        pass

    def build(self, data: Mapping[str, tf.Tensor]) -> tf.keras.Model:
        pass
