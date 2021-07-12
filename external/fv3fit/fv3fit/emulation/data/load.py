import logging
import tensorflow as tf
from typing import Callable, Sequence


logger = logging.getLogger(__name__)


def batched_to_tf_dataset(
    batched_source: Sequence, transform: Callable,
) -> tf.data.Dataset:
    """
    Convert a batched data sequence into a tensorflow dataset to be used
    for ML model training.

    Args:
        batched_source: A sequence of data items to be included in the
            dataset.
        transform: function to process data items into a tensor-compatible
            result
    """

    def get_generator():
        for batch in batched_source:
            output = transform(batch)
            yield tf.data.Dataset.from_tensor_slices(output)

    peeked = next(get_generator())
    signature = tf.data.DatasetSpec.from_value(peeked)
    tf_ds = tf.data.Dataset.from_generator(get_generator, output_signature=signature)

    # Flat map goes from generating tf_dataset -> generating tensors
    tf_ds = tf_ds.prefetch(tf.data.AUTOTUNE).flat_map(lambda x: x)

    return tf_ds
