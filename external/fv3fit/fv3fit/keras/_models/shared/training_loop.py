import tensorflow as tf
from typing import Any, Iterable, List, Optional, Sequence, Callable
import dataclasses
import logging

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Logs:
    loss: float
    val_loss: Optional[float]
    # this could be extended in the future to contain more metrics, or
    # a Mapping for custom metrics, if we also refactor the collection logic


@dataclasses.dataclass
class EpochResult:
    """
    Attributes:
        epoch: count of epoch (starts at zero)
        batch_logs: metrics of `model.fit` from each batch
        epoch_logs: metrics of `model.fit` from the end of the epoch
    """

    epoch: int
    batch_logs: Sequence[Logs]
    epoch_logs: Logs


def sequence_size(seq):
    n = 0
    for _ in seq:
        n += 1
    return n


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
    shuffle_buffer_size: int = 50_000
    batch_size: int = 16
    in_memory: bool = False

    def fit_loop(
        self,
        model: tf.keras.Model,
        Xy: tf.data.Dataset,
        validation_data: Optional[tf.data.Dataset] = None,
        callbacks: Iterable[Callable[[EpochResult], None]] = (),
    ) -> None:
        """
        Args:
            model: keras model to train
            Xy: Dataset containing samples to be passed to model.fit
            validation_data: passed as `validation_data` argument to `model.fit`
            callbacks: if given, these will be called at the end of each epoch
        """
        fit_kwargs = dict(
            callbacks=[EpochCallback(func) for func in callbacks], epochs=self.epochs
        )
        if self.in_memory:
            n_samples = sequence_size(Xy)
            Xy_fit = next(iter(Xy.batch(n_samples)))
            if validation_data is not None:
                n_val_samples = sequence_size(validation_data)
                validation_fit = next(iter(validation_data.batch(n_val_samples)))
            else:
                validation_fit = None
            model.fit(
                x=Xy_fit[0], y=Xy_fit[1], validation_data=validation_fit, **fit_kwargs
            )
        else:
            Xy_fit = (
                Xy.shuffle(buffer_size=self.shuffle_buffer_size)
                .batch(self.batch_size)
                .prefetch(tf.data.AUTOTUNE)
            )
            if validation_data is not None:
                validation_fit = validation_data.batch(self.batch_size).prefetch(
                    tf.data.AUTOTUNE
                )
            else:
                validation_fit = None
            model.fit(Xy_fit, validation_data=validation_fit, **fit_kwargs)


class EpochCallback(tf.keras.callbacks.History):
    def __init__(self, callback: Callable[[EpochResult], Any]):
        self._callback = callback
        self._batch_logs: List[Logs] = []

    def on_train_batch_end(self, epoch: int, logs=None):
        if logs is None or "loss" not in logs:
            raise NotImplementedError(
                "fv3fit epoch callbacks are hard-coded to require loss values"
            )
        self._batch_logs.append(
            Logs(loss=logs["loss"], val_loss=logs.get("val_loss", None))
        )

    def on_epoch_end(self, epoch: int, logs=None):
        if logs is None or "loss" not in logs:
            raise NotImplementedError(
                "fv3fit epoch callbacks are hard-coded to require loss values"
            )
        self._callback(
            EpochResult(
                epoch=epoch,
                batch_logs=tuple(self._batch_logs),
                epoch_logs=Logs(loss=logs["loss"], val_loss=logs.get("val_loss", None)),
            )
        )
        self._batch_logs.clear()
