import tensorflow as tf
from typing import Iterable, Optional, Sequence, Tuple
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

    build_samples: int = 500_000
    Nbatch: int = 1
    n_loop: int = 100
    n_epoch: int = 1

    # epochs: int = 3
    # shuffle_buffer_size: int = 50_000
    # batch_size: int = 16
    # in_memory: bool = True

    # def __post_init__(self):
    #     if self.in_memory:
    #         logger.info(
    #             "training with in_memory=True, if you run out of memory "
    #             "try setting in_memory on TrainingLoopConfig to False"
    #         )
    def fit_loop(config, train_model, inputs, labels, optimizer, get_loss) -> None:
        """
        Args:
            model: keras model to train
            Xy: Dataset containing samples to be passed to model.fit
            validation_data: passed as `validation_data` argument to `model.fit`
            callbacks: if given, these will be called at the end of each epoch
        """
        for epoch in config.n_epoch:  # loop over the dataset multiple times
            for step in range(0, config.n_loop - config.Nbatch, config.Nbatch):
                optimizer.zero_grad()
                loss = get_loss(train_model, inputs, labels)
                loss.backward()
                optimizer.step()
        torch.save(net.state_dict(), WeightsFile + ".pt")
