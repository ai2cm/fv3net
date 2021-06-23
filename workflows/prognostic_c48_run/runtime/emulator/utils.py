import tensorflow as tf


def _get_argsin(levels, n=10):
    shape = (n, levels)
    return (
        tf.random.uniform(shape, -50, 50, dtype=tf.float32),
        tf.random.uniform(shape, -50, 50, dtype=tf.float32),
        tf.random.uniform(shape, 273, 300, dtype=tf.float32),  # t
        tf.random.uniform(shape, 0, 0.01, dtype=tf.float32),  # q
        tf.random.uniform(shape, 100, 110, dtype=tf.float32),  # dp
        tf.random.uniform(shape, 100, 100, dtype=tf.float32),  # dz
    )
