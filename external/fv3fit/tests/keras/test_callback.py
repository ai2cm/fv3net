from dataclasses import dataclass
from typing import Mapping, Sequence
from fv3fit.keras._models.shared.callbacks import TrainingLoopLossHistory, EpochResult


@dataclass
class KerasHistoryMock:
    history: Mapping[str, Sequence[float]]


def test_TrainingLoopLossHistory_callback():
    train_losses = [[1.0, 2.0], [3.0, 4.0]]
    val_losses = [[-1.0, -2.0], [-3.0, -4.0]]

    loss_history = TrainingLoopLossHistory()
    for i, (epoch_train_losses, epoch_val_losses) in enumerate(
        zip(train_losses, val_losses)
    ):
        batch_logs = [
            {"loss": train_loss, "val_loss": val_loss}
            for train_loss, val_loss in zip(epoch_train_losses, epoch_val_losses)
        ]
        epoch_logs = batch_logs[-1]
        result = EpochResult(epoch=i, batch_logs=batch_logs, epoch_logs=epoch_logs)
        loss_history.callback(result)

    assert loss_history.train_loss_end_of_epoch == [2.0, 4.0]
    assert loss_history.val_loss_end_of_epoch == [-2.0, -4.0]
    assert loss_history.train_loss_all == [1.0, 2.0, 3.0, 4.0]
    assert loss_history.val_loss_all == [-1.0, -2.0, -3.0, -4.0]
