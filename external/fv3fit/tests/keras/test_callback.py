from dataclasses import dataclass
from typing import Mapping, Sequence
from fv3fit.keras._models.shared.callbacks import TrainingLoopLossHistory


@dataclass
class KerasHistoryMock:
    history: Mapping[str, Sequence[float]]


@dataclass
class EpochResultMock:
    history: Sequence[KerasHistoryMock]


def _epoch_history(train_losses, val_losses):
    history_over_batches = [
        KerasHistoryMock(history={"loss": [train_loss], "val_loss": [val_loss]})
        for train_loss, val_loss in zip(train_losses, val_losses)
    ]
    return EpochResultMock(history=tuple(history_over_batches))


def test_TrainingLoopLossHistory_callback():
    train_losses = [[1.0, 2.0], [3.0, 4.0]]
    val_losses = [[-1.0, -2.0], [-3.0, -4.0]]

    loss_history = TrainingLoopLossHistory()
    for epoch_train_losses, epoch_val_losses in zip(train_losses, val_losses):
        loss_history.callback(_epoch_history(epoch_train_losses, epoch_val_losses))

    assert loss_history.train_loss_end_of_epoch == [2.0, 4.0]
    assert loss_history.val_loss_end_of_epoch == [-2.0, -4.0]
    assert loss_history.train_loss_all == [1.0, 2.0, 3.0, 4.0]
    assert loss_history.val_loss_all == [-1.0, -2.0, -3.0, -4.0]
