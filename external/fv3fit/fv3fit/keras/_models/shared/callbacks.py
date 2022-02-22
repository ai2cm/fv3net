import numpy as np

from .training_loop import EpochResult

import logging

logger = logging.getLogger(__name__)


class TrainingLoopLossHistory:
    """Summarizes the history across epochs and batches

    Attributes:
        train_loss_end_of_epoch: training loss recorded at end of each epoch
            (i.e. after the last batch iterated upon in the epoch)
        val_loss_end_of_epoch: validation loss recorded at end of each epoch
        train_loss_all: training loss after each iteration over batches in epochs
        val_loss_all: validation loss after each iteration over batches in epochs
    """

    def __init__(self):

        self.train_loss_end_of_epoch = []
        self.val_loss_end_of_epoch = []
        self.train_loss_all = []
        self.val_loss_all = []

    def callback(self, epoch_result: EpochResult):
        self.train_loss_end_of_epoch += epoch_result.history[-1].history["loss"]
        self.val_loss_end_of_epoch += epoch_result.history[-1].history.get(
            "val_loss", [np.nan]
        )
        for batch in epoch_result.history:
            self.train_loss_all += batch.history["loss"]
            self.val_loss_all += batch.history.get("val_loss", [])

    def log_summary(self):
        logger.info(f"All batches train loss history: {self.train_loss_all}")
        logger.info(f"All batches Validation loss history: {self.val_loss_all}")
        logger.info(f"End of epoch train loss history: {self.train_loss_end_of_epoch}")
        logger.info(
            f"End of epoch validation loss history: {self.val_loss_end_of_epoch}"
        )
