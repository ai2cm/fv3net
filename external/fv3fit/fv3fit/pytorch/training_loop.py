import dataclasses
import logging
from typing import Any, Callable, Optional
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

        def evaluate_on_batch(batch_state, model):
            batch_input = torch.as_tensor(batch_state[0]).float().to(DEVICE)
            batch_output = torch.as_tensor(batch_state[1]).float().to(DEVICE)
            loss: torch.Tensor = loss_config.loss(model(batch_input), batch_output)
            return loss

        return _train_loop(
            model=train_model,
            train_data=train_data,
            validation_data=validation_data,
            evaluate_on_batch=evaluate_on_batch,
            optimizer=optimizer,
            n_epoch=self.n_epoch,
            shuffle_buffer_size=self.shuffle_buffer_size,
            samples_per_batch=self.samples_per_batch,
            validation_batch_size=self.validation_batch_size,
        )


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
    shuffle_buffer_size: int = 50_000
    samples_per_batch: int = 1
    save_path: str = "weight.pt"
    multistep: int = 1
    validation_batch_size: Optional[int] = None

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

        def evaluate_on_batch(batch_state, model):
            batch_state = torch.as_tensor(batch_state).float().to(DEVICE)
            loss: torch.Tensor = evaluate_model(
                batch_state=batch_state,
                model=train_model,
                multistep=self.multistep,
                loss=loss_config.loss,
            )
            return loss

        return _train_loop(
            model=train_model,
            train_data=train_data,
            validation_data=validation_data,
            evaluate_on_batch=evaluate_on_batch,
            optimizer=optimizer,
            n_epoch=self.n_epoch,
            shuffle_buffer_size=self.shuffle_buffer_size,
            samples_per_batch=self.samples_per_batch,
            validation_batch_size=self.validation_batch_size,
        )


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


def _train_loop(
    model: torch.nn.Module,
    train_data: tf.data.Dataset,
    validation_data: tf.data.Dataset,
    evaluate_on_batch: Callable[[Any, torch.nn.Module], torch.Tensor],
    optimizer: torch.optim.Optimizer,
    n_epoch: int,
    shuffle_buffer_size: int,
    samples_per_batch: int,
    validation_batch_size: Optional[int] = None,
):

    train_data = train_data.shuffle(buffer_size=shuffle_buffer_size).batch(
        samples_per_batch
    )
    train_data = tfds.as_numpy(train_data)
    if validation_data is not None:
        if validation_batch_size is None:
            validation_batch_size = sequence_size(validation_data)
        validation_data = validation_data.batch(validation_batch_size)
        validation_data = tfds.as_numpy(validation_data)
        min_val_loss = np.inf
        best_weights = None
    for i in range(1, n_epoch + 1):  # loop over the dataset multiple times
        logger.info("starting epoch %d", i)
        train_model = model.train()
        train_losses = []
        for batch_state in train_data:
            optimizer.zero_grad()
            loss = evaluate_on_batch(batch_state, train_model)
            loss.backward()
            train_losses.append(loss)
            optimizer.step()
        train_loss = torch.mean(torch.stack(train_losses))
        logger.info("train loss: %f", train_loss)
        if validation_data is not None:
            val_model = model.eval()
            val_losses = []
            for batch_state in validation_data:
                with torch.no_grad():
                    val_losses.append(evaluate_on_batch(batch_state, val_model))
            val_loss = torch.mean(torch.stack(val_losses))
            logger.info("val_loss %f", val_loss)
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                best_weights = train_model.state_dict()
    if validation_data is not None:
        train_model.load_state_dict(best_weights)
