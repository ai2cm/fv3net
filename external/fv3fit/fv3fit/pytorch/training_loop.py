import dataclasses
import logging
from typing import Callable, Optional
import numpy as np
import torch
import tensorflow_datasets as tfds
from .system import DEVICE
import tensorflow as tf
from fv3fit.tfdataset import sequence_size
from .loss import LossConfig

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class TrainingConfig:
    """
    Training configuration.

    Attributes:
        epochs: number of times to run through the batches when training
        shuffle_buffer_size: size of buffer to use when shuffling samples
        samples_per_batch: number of samples to use in each training batch
        save_path: name of the file to save the best weights
        validation_batch_size: if given, process validation data in batches
            of this size, otherwise process it all at once
    """

    n_epoch: int = 20
    shuffle_buffer_size: int = 10
    samples_per_batch: int = 1
    save_path: str = "weight.pt"
    validation_batch_size: Optional[int] = None

    def fit_loop(
        self,
        train_model: torch.nn.Module,
        train_data: tf.data.Dataset,
        validation_data: tf.data.Dataset,
        optimizer: torch.optim.Optimizer,
        loss_config: LossConfig,
    ) -> None:
        """
        Args:
            train_model: pytorch model to train
            train_data: training dataset containing samples to be passed to the model,
                samples should be tuples with two tensors corresponding to the model
                input and output
            validation_data: validation dataset containing samples to be passed
                to the model, samples should be tuples with two tensors
                corresponding to the model input and output
            optimizer: type of optimizer for the model
            loss_config: configuration of loss function
        """
        train_data = train_data.shuffle(buffer_size=self.shuffle_buffer_size).batch(
            self.samples_per_batch
        )
        train_data = tfds.as_numpy(train_data)
        if validation_data is not None:
            if self.validation_batch_size is None:
                validation_batch_size = sequence_size(validation_data)
            else:
                validation_batch_size = self.validation_batch_size
            validation_data = validation_data.batch(validation_batch_size)
            validation_data = tfds.as_numpy(validation_data)
            min_val_loss = np.inf
            best_weights = None
        for i in range(1, self.n_epoch + 1):  # loop over the dataset multiple times
            logger.info("starting epoch %d", i)
            train_model = train_model.train()
            train_losses = []
            for batch_state in train_data:
                batch_input = torch.as_tensor(batch_state[0]).float().to(DEVICE)
                batch_output = torch.as_tensor(batch_state[1]).float().to(DEVICE)
                optimizer.zero_grad()
                loss: torch.Tensor = loss_config.loss(
                    train_model(batch_input), batch_output
                )
                loss.backward()
                train_losses.append(loss)
                optimizer.step()
            train_loss = torch.mean(torch.stack(train_losses))
            logger.info("train loss: %f", train_loss)
            if validation_data is not None:
                val_model = train_model.eval()
                val_losses = []
                for batch_state in validation_data:
                    batch_input = torch.as_tensor(batch_state[0]).float().to(DEVICE)
                    batch_output = torch.as_tensor(batch_state[1]).float().to(DEVICE)
                    with torch.no_grad():
                        val_losses.append(
                            loss_config.loss(val_model(batch_input), batch_output)
                        )
                val_loss = torch.mean(torch.stack(val_losses))
                logger.info("val_loss %f", val_loss)
                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    best_weights = train_model.state_dict()
        if validation_data is not None:
            train_model.load_state_dict(best_weights)


@dataclasses.dataclass
class AutoregressiveTrainingConfig:
    """
    Training configuration for autoregressive models.

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
        train_model: torch.nn.Module,
        train_data: tf.data.Dataset,
        validation_data: Optional[tf.data.Dataset],
        optimizer: torch.optim.Optimizer,
        loss_config: LossConfig,
    ) -> None:
        """
        Args:
            train_model: pytorch model to train
            train_data: training dataset containing samples to be passed to the model,
                should have dimensions [sample, time, tile, x, y, z]
            validation_data: validation dataset containing samples to be passed
                to the model, should have dimensions [sample, time, tile, x, y, z]
            optimizer: type of optimizer for the model
            loss_config: configuration of loss function
        """
        train_data = (
            train_data.unbatch()
            .shuffle(buffer_size=self.buffer_size)
            .batch(self.samples_per_batch)
        )
        train_data = tfds.as_numpy(train_data)
        if validation_data is not None:
            validation_data = validation_data.unbatch()
            n_validation = sequence_size(validation_data)
            validation_state = (
                torch.as_tensor(next(iter(validation_data.batch(n_validation))).numpy())
                .float()
                .to(DEVICE)
            )
            min_val_loss = np.inf
        else:
            validation_state = None
        for _ in range(1, self.n_epoch + 1):  # loop over the dataset multiple times
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
            if validation_state is not None:
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
