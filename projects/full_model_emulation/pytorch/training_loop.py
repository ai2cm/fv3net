import tensorflow as tf
from typing import Iterable, Optional, Sequence, Tuple
import dataclasses
import logging
import torch

logger = logging.getLogger(__name__)


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
    savemodelpath: str = 'weight.pt'
    def fit_loop(config, train_model, inputs, validation, labels, optimizer, get_loss) -> None:
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
                val_loss = evaluate_model(train_model, loss, validation)
                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    torch.save(train_model.state_dict(), config.savemodelpath)

        #torch.save(net.state_dict(), WeightsFile + ".pt")
