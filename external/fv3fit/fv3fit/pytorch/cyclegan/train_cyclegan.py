import random
from fv3fit._shared.hyperparameters import Hyperparameters
import dataclasses
import tensorflow as tf
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
    """
    Hyperparameters for CycleGAN training.

    Attributes:
        state_variables: list of variables to be transformed by the model
        normalization_fit_samples: number of samples to use when fitting the
            normalization
        network: configuration for the CycleGAN network
        training: configuration for the CycleGAN training
    """

    state_variables: List[str]
    normalization_fit_samples: int = 50_000
    network: "CycleGANNetworkConfig" = dataclasses.field(
        default_factory=lambda: CycleGANNetworkConfig()
    )
    training: "CycleGANTrainingConfig" = dataclasses.field(
        default_factory=lambda: CycleGANTrainingConfig()
    )

    @property
    def variables(self):
        return tuple(self.state_variables)


@dataclasses.dataclass
class CycleGANTrainingConfig:
    """
    Attributes:
        n_epoch: number of epochs to train for
        shuffle_buffer_size: number of samples to use for shuffling the training data
        samples_per_batch: number of samples to use per batch
        validation_batch_size: number of samples to use per batch for validation,
            does not affect training result but allows the use of out-of-sample
            validation data
        in_memory: if True, load the entire dataset into memory as pytorch tensors
            before training. Batches will be statically defined but will be shuffled
            between epochs.
    """

    n_epoch: int = 20
    shuffle_buffer_size: int = 10
    samples_per_batch: int = 1
    validation_batch_size: Optional[int] = None
    in_memory: bool = False

    def fit_loop(
        self,
        train_model: CycleGANTrainer,
        train_data: tf.data.Dataset,
        validation_data: Optional[tf.data.Dataset],
    ) -> None:
        """
        Args:
            train_model: Cycle-GAN to train
            train_data: training dataset containing samples to be passed to the model,
                should be unbatched and have dimensions [time, tile, z, x, y]
            validation_data: validation dataset containing samples to be passed
                to the model, should be unbatched and have dimensions
                [time, tile, z, x, y]
        """
        if self.shuffle_buffer_size > 1:
            train_data = train_data.shuffle(buffer_size=self.shuffle_buffer_size)
        train_data = train_data.batch(self.samples_per_batch)
        train_data_numpy = tfds.as_numpy(train_data)
        if validation_data is not None:
            if self.validation_batch_size is None:
                validation_batch_size = sequence_size(validation_data)
            else:
                validation_batch_size = self.validation_batch_size
            validation_data = validation_data.batch(validation_batch_size)
            validation_data = tfds.as_numpy(validation_data)
        if self.in_memory:
            self._fit_loop_tensor(train_model, train_data_numpy, validation_data)
        else:
            self._fit_loop_dataset(train_model, train_data_numpy, validation_data)

    def _fit_loop_dataset(
        self,
        train_model: CycleGANTrainer,
        train_data_numpy,
        validation_data: Optional[tf.data.Dataset],
    ):
        for i in range(1, self.n_epoch + 1):
            logger.info("starting epoch %d", i)
            train_losses = []
            for batch_state in train_data_numpy:
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

    def _fit_loop_tensor(
        self,
        train_model: CycleGANTrainer,
        train_data_numpy: tf.data.Dataset,
        validation_data: Optional[tf.data.Dataset],
    ):
        train_states = []
        batch_state: Tuple[np.ndarray, np.ndarray]
        for batch_state in train_data_numpy:
            state_a = torch.as_tensor(batch_state[0]).float().to(DEVICE)
            state_b = torch.as_tensor(batch_state[1]).float().to(DEVICE)
            train_states.append((state_a, state_b))
        for i in range(1, self.n_epoch + 1):
            logger.info("starting epoch %d", i)
            train_losses = []
            for state_a, state_b in train_states:
                train_losses.append(train_model.train_on_batch(state_a, state_b))
            random.shuffle(train_states)
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
    # apply_to_tuple(apply_to_mapping(func)), so we do it manually
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
    # [batch, time, tile, x, y, z] -> [batch, time, tile, z, x, y]
    return tf.transpose(data, perm=[0, 1, 2, 5, 3, 4])


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
        n_dims=6,  # [batch, sample, tile, x, y, z]
        mapping_scale_funcs=mapping_scale_funcs,
    )

    if validation_batches is not None:
        val_state = validation_batches.map(get_Xy)
    else:
        val_state = None

    train_state = train_batches.map(get_Xy)

    sample: tf.Tensor = next(iter(train_state))[0]
    train_model = hyperparameters.network.build(
        nx=sample.shape[-3],
        ny=sample.shape[-2],
        n_state=sample.shape[-1],
        n_batch=hyperparameters.training.samples_per_batch,
        state_variables=hyperparameters.state_variables,
        scalers=scalers,
    )

    # time and tile dimensions aren't being used yet while we're using single-tile
    # convolution without a motion constraint, but they will be used in the future

    # MPS backend has a bug where it doesn't properly read striding information when
    # doing 2d convolutions, so we need to use a channels-first data layout
    # from the get-go and do transformations before and after while in numpy/tf space.
    train_state = train_state.map(apply_to_tuple(channels_first))
    if validation_batches is not None:
        val_state = val_state.map(apply_to_tuple(channels_first))

    # batching from the loader is undone here, so we can do our own batching
    # in fit_loop
    train_state = train_state.unbatch()
    if validation_batches is not None:
        val_state = val_state.unbatch()

    hyperparameters.training.fit_loop(
        train_model=train_model, train_data=train_state, validation_data=val_state,
    )
    return train_model.cycle_gan
