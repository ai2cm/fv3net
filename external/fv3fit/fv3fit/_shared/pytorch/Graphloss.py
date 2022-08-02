import torch
import dataclasses
import numpy as np


def _standard_mse():
    loss = torch.nn.MSELoss()

    def custom_loss(y_true, y_pred):
        return loss(y_pred, y_true)

    return custom_loss


def _standard_mae():
    loss = torch.nn.L1Loss()

    def custom_loss(y_true, y_pred):
        return loss(y_true, y_pred)

    return custom_loss


@dataclasses.dataclass
class LossConfig:
    """
    Attributes:
        loss_type: one of "mse" or "mae"
        multistep: number of successive loss calculation before each optimization
    """

    multistep: int = 1
    loss_type: str = "mse"

    def __post_init__(self):
        if self.loss_type not in ("mse", "mae"):
            raise ValueError(
                f"loss_type must be 'mse' or 'mae', got '{self.loss_type}'"
            )

    def loss(self):
        """
        Returns the loss function described by the configuration.

        Args:
            std: standard deviation of the output features

        Returns:
            loss: pytorch loss function
        """
        if self.loss_type == "mse":
            loss = _standard_mse()
        elif self.loss_type == "mae":
            loss = _standard_mae()
        else:
            raise NotImplementedError(f"loss_type {self.loss_type} is not implemented")
        return loss
