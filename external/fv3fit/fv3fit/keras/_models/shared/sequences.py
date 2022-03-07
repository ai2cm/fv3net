import logging
import threading
import queue

from .clip import ClipConfig
import xarray as xr
import numpy as np
import tensorflow as tf
from typing import Sequence, Tuple, List, Any, Optional
from .halos import append_halos
import pace.util
import vcm.safe

from fv3fit._shared.packer import ArrayPacker
from fv3fit._shared.stacking import (
    stack,
    check_empty,
    preserve_samples_per_batch,
    shuffled,
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
    the responsibilities of StackedBatches, unlike XyArraySequence,
    and will append halos to X inputs as requested.
    """

    def __init__(
        self,
        X_names: Sequence[str],
        y_names: Sequence[str],
        dataset_sequence: Sequence[xr.Dataset],
        unstacked_dims=pace.util.Z_DIMS,
        n_halo: int = 0,
        output_clip_config: Optional[ClipConfig] = None,
    ):
        """
        Args:
            X_names: names of input variables
            y_names: names of output variables
            dataset_sequence: sequence of datasets containing
                nput and output variables
            unstacked_dims: dimensions which should be present in the X and y
                arrays apart from the "sample" (stacked) dimension
            n_halo: number of halo points to append to input variables, if given
                then the data must contain "x" and "y" dimensions
            output_clip_config: If provided, will clip levels of output
                variables that are clipped in the clip config when returning arrays.
                Note that this will not affect input variables, even if they are in
                the clip config, as we want to input them in their full length when
                training (even if they are clipped in a subsequent layer).
        """
        horizontal_unstacked_dims = set(pace.util.HORIZONTAL_DIMS).intersection(
            unstacked_dims
        )
        if n_halo > 1 and len(horizontal_unstacked_dims) == 0:
            raise ValueError(
                "when appending halo data (halo > 0), must have "
                "horizontal dimensions in unstacked_dims"
            )
        self.X_names = X_names
        self.y_names = y_names
        self.dataset_sequence = dataset_sequence
        self.n_halo = n_halo
        self.unstacked_dims = unstacked_dims
        self.output_clip_config = output_clip_config

    def __len__(self) -> int:
        return len(self.dataset_sequence)

    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray]:
        ds = self.dataset_sequence[idx]
        X_ds = vcm.safe.get_variables(ds=ds, variables=self.X_names)
        if self.n_halo > 0:
            X_ds = append_halos(X_ds, n_halo=self.n_halo)
        y_ds = vcm.safe.get_variables(ds=ds, variables=self.y_names)
        X_y_datasets = [X_ds, y_ds]
        for i in range(2):
            X_y_datasets[i] = stack(X_y_datasets[i], unstacked_dims=self.unstacked_dims)
            X_y_datasets[i] = check_empty(X_y_datasets[i])
            X_y_datasets[i] = preserve_samples_per_batch(X_y_datasets[i])
        X_y_datasets = shuffled(np.random, X_y_datasets)
        X_ds, y_ds = X_y_datasets
        X = tuple(X_ds[name].values for name in self.X_names)
        y = tuple(y_ds[name].values for name in self.y_names)
        if self.output_clip_config is not None:
            y = tuple(
                self.output_clip_config.clip_along_last_dim(yarr, name)
                for yarr, name in zip(y, self.y_names)
            )
        if np.any([np.any(np.isnan(data)) for data in X]):
            raise ValueError("found NaN in X data")
        if np.any([np.any(np.isnan(data)) for data in y]):
            raise ValueError("found NaN in y data")
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
