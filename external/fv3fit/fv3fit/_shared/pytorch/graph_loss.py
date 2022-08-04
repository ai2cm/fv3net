import torch
import dataclasses


@dataclasses.dataclass
class LossConfig:
    """
    Attributes:
        loss_type: one of "mse" or "mae"
    """

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

    def stepwise_loss(self, multistep, train_model, inputs, labels):
        """
        Multistep loss fucntion, used during training
        Args:
            multistep: number of multistep loss calculation
            train_model: pytoch model
            inputs: input features
            labels: truch to be compared with

        Retunrs:
            average of the losses at each step
        """
        criterion = self.loss()
        sum_loss = 0.0
        # this is just for the identity function,
        # for prediction label would have an index over time
        for step in range(multistep):
            if step == 0:
                outputs = train_model(inputs)
                sum_loss += criterion(outputs, labels)
            else:
                outputs = train_model(outputs)
                sum_loss += criterion(outputs, labels)
        sum_loss = sum_loss / multistep
        return sum_loss