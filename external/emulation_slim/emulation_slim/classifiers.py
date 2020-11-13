import tensorflow as tf


def _less_than_zero_mask(x, to_mask=None):

    non_zero = tf.math.greater_equal(x, tf.zeros_like(x))
    mask = tf.where(non_zero, x=tf.ones_like(x), y=tf.zeros_like(x))

    return mask


def _less_than_equal_zero_mask(x, to_mask=None):

    non_zero = tf.math.greater(x, tf.zeros_like(x))
    mask = tf.where(non_zero, x=tf.ones_like(x), y=tf.zeros_like(x))

    return mask