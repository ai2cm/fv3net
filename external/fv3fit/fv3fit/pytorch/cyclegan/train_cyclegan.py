from fv3fit._shared.hyperparameters import Hyperparameters
import dataclasses
import tensorflow as tf
from fv3fit.pytorch.loss import LossConfig
import torch
from fv3fit.pytorch.system import DEVICE
import tensorflow_datasets as tfds
from fv3fit.tfdataset import sequence_size, apply_to_tuple

from fv3fit._shared import register_training_function
from typing import (
    Callable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)
from fv3fit.tfdataset import ensure_nd
from fv3fit.pytorch.graph.train import get_Xy_map_fn as get_Xy_map_fn_single_domain
from fv3fit._shared.scaler import (
    get_standard_scaler_mapping,
    get_mapping_standard_scale_func,
)
import logging
import numpy as np
from .reloadable import CycleGAN
from .cyclegan_trainer import CycleGANNetworkConfig, CycleGANTrainer

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class CycleGANHyperparameters(Hyperparameters):

    state_variables: List[str]
    normalization_fit_samples: int = 50_000
    network: "CycleGANNetworkConfig" = dataclasses.field(
        default_factory=lambda: CycleGANNetworkConfig()
    )
    training_loop: "CycleGANTrainingConfig" = dataclasses.field(
        default_factory=lambda: CycleGANTrainingConfig()
    )
    loss: LossConfig = LossConfig(loss_type="mse")

    @property
    def variables(self):
        return tuple(self.state_variables)


@dataclasses.dataclass
class CycleGANTrainingConfig:

    n_epoch: int = 20
    shuffle_buffer_size: int = 10
    samples_per_batch: int = 1
    validation_batch_size: Optional[int] = None

    def fit_loop(
        self,
        train_model: CycleGANTrainer,
        train_data: tf.data.Dataset,
        validation_data: Optional[tf.data.Dataset],
    ) -> None:
        """
        Args:
            train_model: cycle-GAN to train
            train_data: training dataset containing samples to be passed to the model,
                should have dimensions [sample, time, tile, x, y, z]
            validation_data: validation dataset containing samples to be passed
                to the model, should have dimensions [sample, time, tile, x, y, z]
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
        for i in range(1, self.n_epoch + 1):
            logger.info("starting epoch %d", i)
            train_losses = []
            for batch_state in train_data:
                state_a = torch.as_tensor(batch_state[0]).float().to(DEVICE)
                state_b = torch.as_tensor(batch_state[1]).float().to(DEVICE)
                train_losses.append(train_model.train_on_batch(state_a, state_b))
            train_loss = {
                name: np.mean([data[name] for data in train_losses])
                for name in train_losses[0]
            }
            logger.info("train_loss: %s", train_loss)

            if validation_data is not None:
                val_loss = train_model.evaluate_on_dataset(validation_data)
                logger.info("val_loss %s", val_loss)


def apply_to_tuple_mapping(func):
    # not sure why, but tensorflow doesn't like parsing
    # apply_to_tuple(apply_to_maping(func)), so we do it manually
    def wrapped(*tuple_of_mapping):
        return tuple(
            {name: func(value) for name, value in mapping.items()}
            for mapping in tuple_of_mapping
        )

    return wrapped


def get_Xy_map_fn(
    state_variables: Sequence[str],
    n_dims: int,  # [batch, time, tile, x, y, z]
    mapping_scale_funcs: Tuple[
        Callable[[Mapping[str, np.ndarray]], Mapping[str, np.ndarray]],
        ...,  # noqa: W504
    ],
):
    funcs = tuple(
        get_Xy_map_fn_single_domain(
            state_variables=state_variables, n_dims=n_dims, mapping_scale_func=func
        )
        for func in mapping_scale_funcs
    )

    def Xy_map_fn(*data: Mapping[str, np.ndarray]):
        return tuple(func(entry) for func, entry in zip(funcs, data))

    return Xy_map_fn


def channels_first(data: tf.Tensor) -> tf.Tensor:
    return tf.transpose(data, perm=[0, 3, 1, 2])


@register_training_function("cyclegan", CycleGANHyperparameters)
def train_cyclegan(
    hyperparameters: CycleGANHyperparameters,
    train_batches: tf.data.Dataset,
    validation_batches: Optional[tf.data.Dataset],
) -> "CycleGAN":
    """
    Train a denoising autoencoder for cubed sphere data.

    Args:
        hyperparameters: configuration for training
        train_batches: training data, as a dataset of Mapping[str, tf.Tensor]
            where each tensor has dimensions [sample, time, tile, x, y(, z)]
        validation_batches: validation data, as a dataset of Mapping[str, tf.Tensor]
            where each tensor has dimensions [sample, time, tile, x, y(, z)]
    """
    train_batches = train_batches.map(apply_to_tuple_mapping(ensure_nd(6)))
    sample_batch = next(
        iter(train_batches.unbatch().batch(hyperparameters.normalization_fit_samples))
    )

    scalers = tuple(get_standard_scaler_mapping(entry) for entry in sample_batch)
    mapping_scale_funcs = tuple(
        get_mapping_standard_scale_func(scaler) for scaler in scalers
    )

    get_Xy = get_Xy_map_fn(
        state_variables=hyperparameters.state_variables,
        n_dims=6,  # [batch, time, tile, x, y, z]
        mapping_scale_funcs=mapping_scale_funcs,
    )

    if validation_batches is not None:
        val_state = validation_batches.map(get_Xy).unbatch()
    else:
        val_state = None

    train_state = train_batches.map(get_Xy).unbatch()

    train_model = hyperparameters.network.build(
        n_state=next(iter(train_state))[0].shape[-1],
        n_batch=hyperparameters.training_loop.samples_per_batch,
        state_variables=hyperparameters.state_variables,
        scalers=scalers,
    )

    # remove time and tile dimensions, while we're using regular convolution
    train_state = train_state.unbatch().map(apply_to_tuple(channels_first)).unbatch()
    if validation_batches is not None:
        val_state = val_state.unbatch().map(apply_to_tuple(channels_first)).unbatch()

    hyperparameters.training_loop.fit_loop(
        train_model=train_model, train_data=train_state, validation_data=val_state,
    )
    return train_model.cycle_gan
