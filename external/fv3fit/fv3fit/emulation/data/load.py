import logging
import os
import tensorflow as tf
from typing import Callable, List,  Sequence

from vcm import get_fs


logger = logging.getLogger(__name__)


def get_nc_files(path: str) -> List[str]:

    fs = get_fs(path)
    files = list(fs.glob(os.path.join(path, "*.nc")))

    return files


def batched_to_tf_dataset(
    batched_source: Sequence,
    transform: Callable,
) -> tf.data.Dataset:

    def get_generator():
        for batch in batched_source:
            output = transform(batch)
            yield tf.data.Dataset.from_tensor_slices(output)

    peeked = next(get_generator())
    signature = tf.data.DatasetSpec.from_value(peeked)
    tf_ds = tf.data.Dataset.from_generator(
        get_generator,
        output_signature=signature
    )

    # Interleave is required with a generator that yields a tf.dataset
    tf_ds = tf_ds.prefetch(tf.data.AUTOTUNE).interleave(lambda x: x)

    return tf_ds
