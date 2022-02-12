import tensorflow as tf
from typing import Iterable, Optional, Sequence, Tuple, Callable
import dataclasses
import numpy as np

from .sequences import ThreadedSequencePreLoader
from loaders.batches import shuffle
import logging


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class EpochResult:
    """
    Attributes:
        epoch: count of epoch (starts at zero)
        history: return value of `model.fit` from each batch
    """

    epoch: int
    history: Sequence[tf.keras.callbacks.History]


@dataclasses.dataclass
class TrainingLoopConfig:
    """
    Attributes:
        epochs: number of times to run through the batches when training
        workers: number of workers for parallelized loading of batches fed into
            training, if 1 uses serial loading instead
        max_queue_size: max number of batches to hold in the parallel loading queue
        batch_size: actual batch_size to pass to keras model.fit,
            independent of number of samples in each data batch in batches
    """

    epochs: int = 3
    workers: int = 1
    max_queue_size: int = 8
    batch_size: int = 16

    def fit_loop(
        self,
        model: tf.keras.Model,
        Xy: Sequence[Tuple[np.ndarray, np.ndarray]],
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        callbacks: Iterable[Callable[[EpochResult], None]] = (),
    ) -> None:
        """
        Args:
            model: keras model to train
            Xy: sequence of data batches to be passed to `model.fit`
            validation_data: passed as `validation_data` argument to `model.fit`
            callbacks: if given, these will be called at the end of each epoch
        """
        for i_epoch in range(self.epochs):
            Xy = shuffle(Xy)
            if self.workers > 1:
                Xy = ThreadedSequencePreLoader(
                    Xy, num_workers=self.workers, max_queue_size=self.max_queue_size
                )
            history = []
            for i_batch, (X, y) in enumerate(Xy):
                logger.info(
                    f"Fitting on batch {i_batch + 1} of {len(Xy)}, "
                    f"of epoch {i_epoch}..."
                )
                history.append(
                    model.fit(
                        X,
                        y,
                        validation_data=validation_data,
                        batch_size=self.batch_size,
                    )
                )
            for callback in callbacks:
                callback(EpochResult(epoch=i_epoch, history=tuple(history)))
