from fv3fit._shared.hyperparameters import Hyperparameters
import dataclasses
import tensorflow as tf
from fv3fit.pytorch.predict import PytorchPredictor
from fv3fit.pytorch.loss import LossConfig
from fv3fit.pytorch.optimizer import OptimizerConfig
import tensorflow_datasets as tfds
from fv3fit.tfdataset import sequence_size
import torch
import numpy as np
from ..system import DEVICE

from fv3fit._shared import register_training_function
from typing import (
    List,
    Optional,
    Tuple,
)
from fv3fit.tfdataset import ensure_nd, apply_to_mapping
from .network import Generator
from fv3fit.pytorch.graph.train import (
    get_scalers,
    get_mapping_scale_func,
    get_Xy_dataset,
)
from toolz import curry
import logging

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class GeneratorConfig:
    n_convolutions: int = 3
    n_resnet: int = 3
    max_filters: int = 256


@dataclasses.dataclass
class AutoencoderHyperparameters(Hyperparameters):

    state_variables: List[str]
    normalization_fit_samples: int = 50_000
    optimizer_config: OptimizerConfig = dataclasses.field(
        default_factory=lambda: OptimizerConfig("AdamW")
    )
    generator: GeneratorConfig = dataclasses.field(
        default_factory=lambda: GeneratorConfig()
    )
    training_loop: "TrainingLoopConfig" = dataclasses.field(
        default_factory=lambda: TrainingLoopConfig()
    )
    loss: LossConfig = LossConfig(loss_type="mse")

    @property
    def variables(self):
        return tuple(self.state_variables)


@dataclasses.dataclass
class TrainingLoopConfig:
    """
    Attributes:
        epochs: number of times to run through the batches when training
        shuffle_buffer_size: size of buffer to use when shuffling samples
        save_path: name of the file to save the best weights
        do_multistep: if True, use multistep loss calculation
        multistep: number of steps in multistep loss calculation
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
                samples should be tuples with two tensors of shape
                [sample, time, tile, x, y, z]
            validation_data: validation dataset containing samples to be passed
                to the model, samples should be tuples with two tensors
                of shape [sample, time, tile, x, y, z]
            optimizer: type of optimizer for the model
            loss_config: configuration of loss function
        """
        train_data = (
            flatten_dims(train_data)
            .shuffle(buffer_size=self.shuffle_buffer_size)
            .batch(self.samples_per_batch)
        )
        train_data = tfds.as_numpy(train_data)
        if validation_data is not None:
            if self.validation_batch_size is None:
                validation_batch_size = sequence_size(validation_data)
            else:
                validation_batch_size = self.validation_batch_size
            validation_data = flatten_dims(validation_data).batch(validation_batch_size)
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


@curry
def define_noisy_input(data: tf.Tensor, stdev=0.1) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Given data, return a tuple with a noisy version of the data and the original data.
    """
    noisy = data + tf.random.normal(shape=tf.shape(data), stddev=stdev)
    return (noisy, data)


def flatten_dims(dataset: tf.data.Dataset) -> tf.data.Dataset:
    """Transform [batch, time, tile, x, y, z] to [sample, x, y, z]"""
    return dataset.unbatch().unbatch().unbatch()


@register_training_function("autoencoder", AutoencoderHyperparameters)
def train_autoencoder(
    hyperparameters: AutoencoderHyperparameters,
    train_batches: tf.data.Dataset,
    validation_batches: Optional[tf.data.Dataset],
) -> PytorchPredictor:
    """
    Train a denoising autoencoder for cubed sphere data.

    Args:
        hyperparameters: configuration for training
        train_batches: training data, as a dataset of Mapping[str, tf.Tensor]
            where each tensor has dimensions [sample, time, tile, x, y(, z)]
        validation_batches: validation data, as a dataset of Mapping[str, tf.Tensor]
            where each tensor has dimensions [sample, time, tile, x, y(, z)]
    """
    train_batches = train_batches.map(apply_to_mapping(ensure_nd(6)))
    sample_batch = next(
        iter(train_batches.unbatch().batch(hyperparameters.normalization_fit_samples))
    )

    scalers = get_scalers(sample_batch)
    mapping_scale_func = get_mapping_scale_func(scalers)

    get_state = curry(get_Xy_dataset)(
        state_variables=hyperparameters.state_variables,
        n_dims=6,  # [batch, time, tile, x, y, z]
        mapping_scale_func=mapping_scale_func,
    )

    if validation_batches is not None:
        val_state = get_state(data=validation_batches)
    else:
        val_state = None

    train_state = get_state(data=train_batches)

    train_model = build_model(
        hyperparameters.generator, n_state=next(iter(train_state)).shape[-1]
    )
    print(train_model)
    optimizer = hyperparameters.optimizer_config

    train_state = train_state.map(define_noisy_input(stdev=0.5))
    if validation_batches is not None:
        val_state = val_state.map(define_noisy_input(stdev=0.5))

    hyperparameters.training_loop.fit_loop(
        train_model=train_model,
        train_data=train_state,
        validation_data=val_state,
        optimizer=optimizer.instance(train_model.parameters()),
        loss_config=hyperparameters.loss,
    )

    predictor = PytorchPredictor(
        input_variables=hyperparameters.state_variables,
        output_variables=hyperparameters.state_variables,
        model=train_model,
        scalers=scalers,
    )
    return predictor


def build_model(config: GeneratorConfig, n_state: int) -> Generator:
    return Generator(
        channels=n_state,
        n_convolutions=config.n_convolutions,
        n_resnet=config.n_resnet,
        max_filters=config.max_filters,
    )
