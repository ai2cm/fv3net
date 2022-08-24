from fv3fit._shared.hyperparameters import Hyperparameters
import random
import dataclasses
import tensorflow as tf
from fv3fit.pytorch.predict import PytorchPredictor
from fv3fit.pytorch.loss import LossConfig
from fv3fit.pytorch.optimizer import OptimizerConfig
import torch
from fv3fit.pytorch.system import DEVICE
import tensorflow_datasets as tfds
from fv3fit.tfdataset import sequence_size

from fv3fit._shared import register_training_function
from typing import (
    Dict,
    List,
    Optional,
)
from fv3fit.tfdataset import ensure_nd, apply_to_mapping
from .network import Discriminator, Generator, GeneratorConfig, DiscriminatorConfig
from fv3fit.pytorch.graph.train import (
    get_scalers,
    get_mapping_scale_func,
    get_Xy_dataset,
)
from toolz import curry
import logging
import numpy as np

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class CycleGANHyperparameters(Hyperparameters):

    state_variables: List[str]
    normalization_fit_samples: int = 50_000
    optimizer_config: OptimizerConfig = dataclasses.field(
        default_factory=lambda: OptimizerConfig("AdamW")
    )
    generator: GeneratorConfig = dataclasses.field(
        default_factory=lambda: GeneratorConfig()
    )
    discriminator: DiscriminatorConfig = dataclasses.field(
        default_factory=lambda: DiscriminatorConfig()
    )
    training_loop: "CycleGANTrainingConfig" = dataclasses.field(
        default_factory=lambda: CycleGANTrainingConfig()
    )
    loss: LossConfig = LossConfig(loss_type="mse")

    @property
    def variables(self):
        return tuple(self.state_variables)


def flatten_dims(dataset: tf.data.Dataset) -> tf.data.Dataset:
    """Transform [batch, time, tile, x, y, z] to [sample, x, y, z]"""
    return dataset.unbatch().unbatch().unbatch()


class CycleGANTrainingConfig:

    n_epoch: int = 20
    shuffle_buffer_size: int = 10
    samples_per_batch: int = 1
    validation_batch_size: Optional[int] = None

    def fit_loop(
        self,
        train_model: "CycleGAN",
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
                train_losses.append(train_model.train_on_batch(*batch_state))
            train_loss = torch.mean(torch.stack(train_losses))
            logger.info("train_loss: %f", train_loss)
            if validation_data is not None:
                val_loss = train_model.evaluate_on_dataset(validation_data)
                logger.info("val_loss %f", val_loss)


@register_training_function("cyclegan", CycleGANHyperparameters)
def train_cyclegan(
    hyperparameters: CycleGANHyperparameters,
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

    train_state = flatten_dims(train_state)
    if validation_batches is not None:
        val_state = flatten_dims(val_state)

    hyperparameters.training_loop.fit_loop(
        train_model=train_model, train_data=train_state, validation_data=val_state,
    )

    predictor = PytorchPredictor(
        input_variables=hyperparameters.state_variables,
        output_variables=hyperparameters.state_variables,
        model=train_model,
        scalers=scalers,
    )
    return predictor


class ReplayBuffer:

    # To reduce model oscillation during training, we update the discriminator
    # using a history of generated data instead of the most recently generated data
    # according to Shrivastava et al. (2017).

    def __init__(self, max_size=50):
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data: torch.Tensor) -> torch.autograd.Variable:
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.autograd.Variable(torch.cat(to_return))


class StatsCollector:
    def __init__(self, n_dims_keep: int):
        self.n_dims_keep = n_dims_keep
        self._sum = np.asarray(0.0, dtype=np.float64)
        self._sum_squared = np.asarray(0.0, dtype=np.float64)
        self._count = 0

    def observe(self, data: np.ndarray):
        mean_dims = tuple(range(0, len(data.shape) - self.n_dims_keep))
        data = data.astype(np.float64)
        self._sum += data.mean(dims=mean_dims)
        self._sum_squared += (data ** 2).mean(dims=mean_dims)
        self._count += 1

    @property
    def mean(self) -> np.ndarray:
        return self._sum / self._count

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self._sum_squared / self._count - self.mean() ** 2)


def get_r2(predicted, target) -> float:
    """
    Compute the R^2 statistic for the predicted and target data.
    """
    return (
        1.0
        - ((target - predicted) ** 2).mean() / ((target - target.mean()) ** 2).mean()
    )


class CycleGAN:

    # This class based loosely on
    # https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/cycle_gan_model.py

    # Copyright Facebook, BSD license
    # https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/c99ce7c4e781712e0252c6127ad1a4e8021cc489/LICENSE

    generator_a_to_b: Generator
    generator_b_to_a: Generator
    discriminator_a: Discriminator
    discriminator_b: Discriminator
    optimizer_generator: torch.optim.Optimizer
    optimizer_discriminator: torch.optim.Optimizer
    identity_loss: torch.nn.Module
    cycle_loss: torch.nn.Module
    gan_loss: torch.nn.Module
    batch_size: int
    identity_weight: float = 0.5
    cycle_weight: float = 1.0
    gan_weight: float = 1.0

    def __post_init__(self):
        self.target_real = torch.autograd.Variable(
            torch.Tensor(self.batch_size).fill_(1.0), requires_grad=False
        )
        self.target_fake = torch.autograd.Variable(
            torch.Tensor(self.batch_size).fill_(0.0), requires_grad=False
        )
        self.fake_a_buffer = ReplayBuffer()
        self.fake_b_buffer = ReplayBuffer()

    def evaluate_on_dataset(
        self, dataset: tf.data.Dataset, n_dims_keep: int = 4
    ) -> Dict[str, float]:
        stats_real_a = StatsCollector(n_dims_keep)
        stats_real_b = StatsCollector(n_dims_keep)
        stats_gen_a = StatsCollector(n_dims_keep)
        stats_gen_b = StatsCollector(n_dims_keep)
        real_a: np.ndarray
        real_b: np.ndarray
        for real_a, real_b in dataset:
            stats_real_a.observe(real_a)
            stats_real_b.observe(real_b)
            gen_a: torch.Tensor = self.generator_a_to_b(
                torch.as_tensor(real_a).float().to(DEVICE)
            )
            gen_b: torch.Tensor = self.generator_b_to_a(
                torch.as_tensor(real_b).float().to(DEVICE)
            )
            stats_gen_a.observe(gen_a.detach().cpu().numpy())
            stats_gen_b.observe(gen_b.detach().cpu().numpy())
        metrics = {
            "r2_mean_a": get_r2(stats_real_a.mean, stats_gen_a.mean),
            "bias_mean_a": np.mean(stats_real_a.mean - stats_gen_a.mean),
            "r2_mean_b": get_r2(stats_real_b.mean, stats_gen_b.mean),
            "bias_mean_b": np.mean(stats_real_b.mean - stats_gen_b.mean),
            "r2_std_a": get_r2(stats_real_a.std, stats_gen_a.std),
            "bias_std_a": np.mean(stats_real_a.std - stats_gen_a.std),
            "r2_std_b": get_r2(stats_real_b.std, stats_gen_b.std),
            "bias_std_b": np.mean(stats_real_b.std - stats_gen_b.std),
        }
        return metrics

    def train_on_batch(self, real_a: torch.Tensor, real_b: torch.Tensor) -> float:
        fake_b = self.generator_a_to_b(real_a)
        fake_a = self.generator_b_to_a(real_b)
        reconstructed_a = self.generator_b_to_a(fake_b)
        reconstructed_b = self.generator_a_to_b(fake_a)

        # Generators A2B and B2A ######

        # don't update discriminators when training generators to fool them
        set_requires_grad(
            [self.discriminator_a, self.discriminator_b], requires_grad=False
        )

        self.optimizer_generator.zero_grad()

        # Identity loss
        # G_A2B(B) should equal B if real B is fed
        same_b = self.generator_a_to_b(real_b)
        loss_identity_b = self.identity_loss(same_b, real_b) * self.identity_weight
        # G_B2A(A) should equal A if real A is fed
        same_a = self.generator_b_to_a(real_b)
        loss_identity_a = self.identity_loss(same_a, real_a) * self.identity_weight

        # GAN loss
        fake_b = self.generator_a_to_b(real_a)
        pred_fake = self.discriminator_b(fake_b)
        loss_gan_a_to_b = self.gan_loss(pred_fake, self.target_real)

        fake_A = self.generator_b_to_a(real_b)
        pred_fake = self.discriminator_a(fake_A)
        loss_gan_b_to_a = self.gan_loss(pred_fake, self.target_real)

        # Cycle loss
        loss_cycle_a_b_a = self.cycle_loss(reconstructed_a, real_a) * self.cycle_weight
        loss_cycle_b_a_b = self.cycle_loss(reconstructed_b, real_b) * self.cycle_weight

        # Total loss
        loss_g: torch.Tensor = (
            loss_identity_a
            + loss_identity_b
            + loss_gan_a_to_b
            + loss_gan_b_to_a
            + loss_cycle_a_b_a
            + loss_cycle_b_a_b
        )
        loss_g.backward()

        self.optimizer_generator.step()

        # Discriminators A and B ######

        # do update discriminators when training them to identify samples
        set_requires_grad(
            [self.discriminator_a, self.discriminator_b], requires_grad=True
        )

        self.optimizer_discriminator.zero_grad()

        # Real loss
        pred_real = self.discriminator_a(real_a)
        loss_d_a_real = self.gan_loss(pred_real, self.target_real)

        # Fake loss
        fake_a = self.fake_a_buffer.push_and_pop(fake_a)
        pred_a_fake = self.discriminator_a(fake_a.detach())
        loss_d_a_fake = self.gan_loss(pred_a_fake, self.target_fake)

        # Real loss
        pred_real = self.discriminator_b(real_b)
        loss_d_b_real = self.gan_loss(pred_real, self.target_real)

        # Fake loss
        fake_b = self.fake_b_buffer.push_and_pop(fake_b)
        pred_b_fake = self.discriminator_b(fake_b.detach())
        loss_d_b_fake = self.gan_loss(pred_b_fake, self.target_fake)

        # Total loss
        loss_d: torch.Tensor = (
            loss_d_b_real + loss_d_b_fake + loss_d_a_real + loss_d_a_fake
        ) * 0.5
        loss_d.backward()

        self.optimizer_discriminator.step()
        return float(loss_g + loss_d)


def build_model(config: GeneratorConfig, n_state: int) -> CycleGAN:
    return Generator(
        channels=n_state,
        n_convolutions=config.n_convolutions,
        n_resnet=config.n_resnet,
        max_filters=config.max_filters,
    )


def set_requires_grad(nets: List[torch.nn.Module], requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad
