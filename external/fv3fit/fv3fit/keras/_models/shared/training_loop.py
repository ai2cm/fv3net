import tensorflow as tf
from typing import Any, Iterable, List, Optional, Sequence, Callable, Tuple
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


def _tfdataset_to_tensor_sequence(
    dataset: tf.data.Dataset,
) -> Tuple[Sequence[tf.Tensor], Sequence[tf.Tensor]]:
    dataset = dataset.unbatch()
    n_samples = sequence_size(dataset)
    tensor_sequence = next(iter(dataset.batch(n_samples)))
    return tensor_sequence


def _shuffle_batched_tfdataset(
    data: tf.data.Dataset, sample_buffer_size: int
) -> tf.data.Dataset:
    """
    Given a tfdataset with a sample dimension in its elements, return a tfdataset
    without that sample dimension which is the result of fully shuffling
    those elements (batches) before doing a per-sample shuffle
    with the given buffer size.
    """
    n_batches = sequence_size(data)
    return (
        data.shuffle(
            n_batches
        )  # elements are [sample, z], sample dimension not shuffled
        .unbatch()  # elements are [z]
        .shuffle(buffer_size=sample_buffer_size)
    )


@dataclasses.dataclass
class TrainingLoopConfig:
    """
    Attributes:
        epochs: number of times to run through the batches when training
        shuffle_buffer_size: size of buffer to use when shuffling samples, only
            applies if in_memory=False
        batch_size: actual batch_size to pass to keras model.fit,
            independent of number of samples in each data batch in batches
        in_memory: if True, cast incoming data to eagerly loaded numpy arrays
            before calling keras fit routine (uses tf.data.Dataset if False).
    """

    epochs: int = 3
    shuffle_buffer_size: int = 50_000
    batch_size: int = 16
    in_memory: bool = True

    def __post_init__(self):
        if self.in_memory:
            logger.info(
                "training with in_memory=True, if you run out of memory "
                "try setting in_memory on TrainingLoopConfig to False"
            )

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
            Xy_fit = _tfdataset_to_tensor_sequence(Xy)
            if validation_data is not None:
                validation_fit: Optional[
                    Tuple[Sequence[tf.Tensor], Sequence[tf.Tensor]]
                ] = _tfdataset_to_tensor_sequence(validation_data)
            else:
                validation_fit = None
            model.fit(
                x=Xy_fit[0],
                y=Xy_fit[1],
                validation_data=validation_fit,
                batch_size=self.batch_size,
                **fit_kwargs,
            )
        else:
            Xy_fit = (  # input elements are [sample, z]
                _shuffle_batched_tfdataset(
                    Xy, sample_buffer_size=self.shuffle_buffer_size
                )  # elements are [z]
                .batch(self.batch_size)  # elements are [sample, z]
                .prefetch(tf.data.AUTOTUNE)
            )
            if validation_data is not None:
                validation_fit = (
                    validation_data.unbatch()
                    .batch(self.batch_size)
                    .prefetch(tf.data.AUTOTUNE)
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
