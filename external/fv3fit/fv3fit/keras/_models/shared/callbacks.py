import numpy as np

from .training_loop import EpochResult

import logging

logger = logging.getLogger(__name__)


class TrainingLoopLossHistory:
    """Summarizes the history across epochs and batches
    *_loss_end: loss at end of epoch
    *_loss_full: loss after each iteration over batches in epochs
    """

    def __init__(self):
        self.train_loss_end_of_epoch = []
        self.val_loss_end_of_epoch = []
        self.train_loss_all = []
        self.val_loss_all = []

    def callback(self, epoch_result: EpochResult):
        for batch in epoch_result.history:
            self.train_loss_end_of_epoch.append(batch.history["loss"][-1])
            self.val_loss_end_of_epoch.append(
                batch.history.get("val_loss", [np.nan])[-1]
            )
            self.train_loss_all += batch.history["loss"]
            self.val_loss_all += batch.history.get("val_loss", [])

    def log_summary(self):
        logger.info(f"All batches train loss history: {self.train_loss_all}")
        logger.info(f"All batches Validation loss history: {self.val_loss_all}")
        logger.info(f"End of epoch train loss history: {self.train_loss_end_of_epoch}")
        logger.info(
            f"End of epoch validation loss history: {self.val_loss_end_of_epoch}"
        )
