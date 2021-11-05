import logging
import threading
import queue
import xarray as xr
import numpy as np
import tensorflow as tf
from typing import Sequence, Tuple, List, Any
from .halos import append_halos
import fv3gfs.util
import vcm.safe

from fv3fit._shared.packer import ArrayPacker
from fv3fit._shared.stacking import (
    stack,
    check_empty,
    preserve_samples_per_batch,
    shuffled,
    SAMPLE_DIM_NAME,
)

logger = logging.getLogger(__name__)


class XyArraySequence(tf.keras.utils.Sequence):
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


class XyMultiArraySequence(tf.keras.utils.Sequence):
    """
    Wrapper object converting a sequence of unstacked batch datasets
    to a stacked, shuffled sequence of tuples of input/output numpy arrays.

    These tuples contain one unpacked numpy array for each input/output,
    in contrast to XyArraySequence which is specialized to the case
    of a single input/output of packed arrays. This class also performs
    the responsibilities of StackedBatches, unlike XyArraySequence
    """

    def __init__(
        self,
        X_names: Sequence[str],
        y_names: Sequence[str],
        dataset_sequence: Sequence[xr.Dataset],
        unstacked_dims=fv3gfs.util.Z_DIMS,
        n_halo: int = 0,
    ):
        self.X_names = X_names
        self.y_names = y_names
        self.dataset_sequence = dataset_sequence
        self.n_halo = n_halo
        self.unstacked_dims = unstacked_dims

    def __len__(self) -> int:
        return len(self.dataset_sequence)

    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray]:
        ds = self.dataset_sequence[idx]
        X_y_datasets = [
            vcm.safe.get_variables(ds=ds, variables=self.X_names),
            vcm.safe.get_variables(ds=ds, variables=self.y_names),
        ]
        X_y_datasets[0] = append_halos(X_y_datasets[0], n_halo=self.n_halo)
        for i in range(2):
            X_y_datasets[i] = stack(
                X_y_datasets[i], unstacked_dims=self.unstacked_dims
            ).dropna(dim=SAMPLE_DIM_NAME)
            X_y_datasets[i] = check_empty(X_y_datasets[i])
            X_y_datasets[i] = preserve_samples_per_batch(X_y_datasets[i])
        X_y_datasets = shuffled(np.random, X_y_datasets)
        X_ds, y_ds = X_y_datasets
        X = tuple(X_ds[name].values for name in self.X_names)
        y = tuple(y_ds[name].values for name in self.y_names)
        return X, y


class ThreadedSequencePreLoader(tf.keras.utils.Sequence):
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
