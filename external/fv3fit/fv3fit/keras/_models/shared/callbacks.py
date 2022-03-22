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
        logger.info(
            "saving end of epoch metrics for epoch "
            f"{epoch_result.epoch}: {epoch_result.epoch_logs}"
        )
        self.train_loss_end_of_epoch.append(epoch_result.epoch_logs.loss)
        self.val_loss_end_of_epoch.append(epoch_result.epoch_logs.val_loss)
        for batch_log in epoch_result.batch_logs:
            self.train_loss_all.append(batch_log.loss)
            self.val_loss_all.append(batch_log.val_loss)

    def log_summary(self):
        logger.info(f"All batches train loss history: {self.train_loss_all}")
        logger.info(f"All batches Validation loss history: {self.val_loss_all}")
        logger.info(f"End of epoch train loss history: {self.train_loss_end_of_epoch}")
        logger.info(
            f"End of epoch validation loss history: {self.val_loss_end_of_epoch}"
        )
