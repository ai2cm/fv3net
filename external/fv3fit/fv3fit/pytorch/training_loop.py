import dataclasses
import logging
from typing import Callable
import numpy as np
import torch
import tensorflow_datasets as tfds
from .system import DEVICE
import tensorflow as tf
from fv3fit.tfdataset import sequence_size

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class TrainingLoopConfig:
    """
    Attributes:
        epochs: number of times to run through the batches when training
        shuffle_buffer_size: size of buffer to use when shuffling samples
        save_path: name of the file to save the best weights
        multistep: number of multistep loss calculation
    """

    n_epoch: int = 20
    buffer_size: int = 50_000
    samples_per_batch: int = 1
    save_path: str = "weight.pt"
    multistep: int = 1

    def fit_loop(
        self,
        loss_config,
        train_model: torch.nn.Module,
        train_data: tf.data.Dataset,
        validation_data: tf.data.Dataset,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """
        Args:
            train_model: pytorch model to train
            train_data: training dataset containing samples to be passed to the model,
                should have dimensions [sample, time, tile, x, y, z]
            validation_data: validation dataset containing samples to be passed
                to the model, should have dimensions [sample, time, tile, x, y, z]
            optimizer: type of optimizer for the model
            get_loss: Multistep loss function
            multistep: number of multi-step loss calculation
        """
        train_data = (
            train_data.unbatch()
            .shuffle(buffer_size=self.buffer_size)
            .batch(self.samples_per_batch)
        )
        train_data = tfds.as_numpy(train_data)
        validation_data = validation_data.unbatch()
        n_validation = sequence_size(validation_data)
        validation_state = (
            torch.as_tensor(next(iter(validation_data.batch(n_validation))).numpy())
            .float()
            .to(DEVICE)
        )
        min_val_loss = np.inf
        for epoch in range(1, self.n_epoch + 1):  # loop over the dataset multiple times
            train_model = train_model.train()
            for batch_state in train_data:
                batch_state = torch.as_tensor(batch_state).float().to(DEVICE)
                optimizer.zero_grad()
                loss = evaluate_model(
                    batch_state=batch_state,
                    model=train_model,
                    multistep=self.multistep,
                    loss=loss_config.loss,
                )
                loss.backward()
                optimizer.step()
            val_model = train_model.eval()
            with torch.no_grad():
                val_loss = evaluate_model(
                    validation_state,
                    model=val_model,
                    multistep=self.multistep,
                    loss=loss_config.loss,
                )
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                torch.save(train_model.state_dict(), self.save_path)


def evaluate_model(
    batch_state: torch.Tensor,
    model: torch.nn.Module,
    multistep: int,
    loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    total_loss = 0.0
    state_snapshot = batch_state[:, 0, :]
    for step in range(multistep):
        state_snapshot = model(state_snapshot)
        target_state = batch_state[:, step + 1, :]
        total_loss += loss(state_snapshot, target_state)
    return total_loss / multistep
