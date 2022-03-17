import gc
import logging
from typing import Optional

import tensorflow as tf

logger = logging.getLogger(__name__)


class GarbageCollectionCallback(tf.keras.callbacks.Callback):
    """Collect garbage"""

    def __init__(self, batch_frequency: Optional[int] = None):
        self.batch_frequency = batch_frequency
        self._count = 0

    def _collect(self):
        logger.debug("Collecting garbage.")
        gc.collect()
        tf.keras.backend.clear_session()
        self._count = 0

    def on_epoch_end(self, epoch, logs=None):
        self._collect()

    def on_batch_begin(self, batch, logs=None):
        self._count += 1
        if (self.batch_frequency is not None) and (self._count >= self.batch_frequency):
            self._collect()
