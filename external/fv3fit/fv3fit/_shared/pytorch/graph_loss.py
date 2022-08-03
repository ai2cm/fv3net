import torch
import dataclasses


@dataclasses.dataclass
class LossConfig:
    """
    Attributes:
        loss_type: one of "mse" or "mae"
        multistep: number of multi-step loss calculation
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
            loss = torch.nn.MSELoss()
        elif self.loss_type == "mae":
            loss = torch.nn.L1Loss()
        else:
            raise NotImplementedError(f"loss_type {self.loss_type} is not implemented")
        return loss
