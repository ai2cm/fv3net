import logging
import threading
import queue
import xarray as xr
import numpy as np
import tensorflow as tf
from typing import Sequence, Tuple, List, Any

from ..._shared import ArrayPacker

logger = logging.getLogger(__name__)


class _XyArraySequence(tf.keras.utils.Sequence):
    """
    Wrapper object converting a sequence of batch datasets
    to a sequence of input/output numpy arrays.
    """

    def __init__(
        self,
        X_packer: ArrayPacker,
        y_packer: ArrayPacker,
        dataset_sequence: Sequence[xr.Dataset],
    ):
        self.X_packer = X_packer
        self.y_packer = y_packer
        self.dataset_sequence = dataset_sequence

    def __len__(self) -> int:
        return len(self.dataset_sequence)

    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray]:
        ds = self.dataset_sequence[idx]
        X = self.X_packer.to_array(ds)
        y = self.y_packer.to_array(ds)
        return X, y


class _TargetToBool(_XyArraySequence):

    """
    Convert y values True where they pass threshold comparison
    """

    def set_y_thresh(self, thresh):
        self.y_thresh = thresh

    def __getitem__(self, item):
        X, y = super().__getitem__(item)
        return X, (abs(y) >= self.y_thresh)


class _BalanceNegativeSkewBinary(tf.keras.utils.Sequence):

    """
    Balance negatively skewed samples assumes single y output
    otherwise we'll need to implement batch adjustments.
    """

    # TODO: Implement these on batches instead

    def __init__(self, xy_sequence: _TargetToBool, min_sample_size=32):
        self._xy_seq = xy_sequence
        self._min_sample_size = min_sample_size

    def __len__(self):
        return len(self._xy_seq)

    def __getitem__(self, item):
        return self._balance_negative_skew(*self._xy_seq[item])

    def _balance_negative_skew(self, X, y):
        y_has_true = y.sum(axis=1) > 0
        num_positive = y_has_true.sum()
        num_negative = len(y_has_true) - num_positive

        if num_negative > num_positive:
            y_all_false = np.logical_not(y_has_true)
            (false_locs,) = zip(*np.argwhere(y_all_false))
            idx_to_keep = np.random.choice(false_locs, size=num_positive, replace=False)
            y_all_false[idx_to_keep] = False
            samples_to_keep = y_has_true | np.logical_not(y_all_false)
        else:
            samples_to_keep = np.ones_like(y_has_true, dtype=np.bool)

        return X[samples_to_keep], y[samples_to_keep]


class _ThreadedSequencePreLoader(tf.keras.utils.Sequence):
    """
    Wrapper object for using a threaded pre-load to provide
    items for a generator.

    Note: This might not preserve strict sequence ordering
        ... but its faster.  Beware that it can load up to
        max_queue_size + num_workers into memory at the
        same time.
    """

    def __init__(
        self,
        seq: tf.keras.utils.Sequence,
        num_workers: int = 4,
        max_queue_size: int = 6,
    ):
        logger.debug(
            f"Initializing threaded batch loader with {num_workers} workers"
            f" and max queue size of {max_queue_size}"
        )
        self._seq = seq
        self.num_workers = num_workers
        self.max_queue_size = max_queue_size

    def __len__(self):
        return len(self._seq)

    def __getitem__(self, index) -> Any:
        return self._seq[index]

    def __iter__(self):

        init_q = queue.Queue()
        for idx in list(range(len(self))):
            init_q.put(idx)

        event = threading.Event()
        preloaded = queue.Queue(maxsize=self.max_queue_size)

        producers: List[threading.Thread] = [
            threading.Thread(
                target=self._produce_loaded_batches, args=(init_q, preloaded, event)
            )
            for i in range(self.num_workers)
        ]

        # Start workers
        for thread in producers:
            thread.start()
            logger.debug(f"Started worker thread {thread.ident}")

        # Generator on preloaded batches
        for i in range(len(self)):
            yield preloaded.get()

        # stop threads
        event.set()
        for thread in producers:
            logger.debug(f"Joining worker thread {thread.ident}")
            thread.join()

    def _produce_loaded_batches(self, src_q, dst_q, event):
        while not event.is_set():

            try:
                item = src_q.get(timeout=5)
            except queue.Empty:
                continue

            dst_q.put(self[item])
            src_q.task_done()
            logger.debug(f"Loadded batch #{item}")
