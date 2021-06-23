import tensorflow as tf
from runtime.emulator.thermo import SpecificHumidityBasis, ThermoBasis


def _get_argsin(levels: int, n: int = 10) -> ThermoBasis:
    shape = (n, levels)
    return SpecificHumidityBasis(
        (
            tf.random.uniform(shape, -50, 50, dtype=tf.float32),
            tf.random.uniform(shape, -50, 50, dtype=tf.float32),
            tf.random.uniform(shape, 273, 300, dtype=tf.float32),  # t
            tf.random.uniform(shape, 0, 0.01, dtype=tf.float32),  # q
            tf.random.uniform(shape, 100, 110, dtype=tf.float32),  # dp
            tf.random.uniform(shape, 100, 100, dtype=tf.float32),  # dz
        )
    )
