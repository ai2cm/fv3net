import random
from fv3fit._shared.hyperparameters import Hyperparameters
import dataclasses
from fv3fit.pytorch.cyclegan.generator import GeneratorConfig
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
class FMRTrainer:
    """
    A trainer for a FMR model.

    Attributes:
        cycle_gan: the CycleGAN model to train
        optimizer_generator: the optimizer for the generator
        optimizer_discriminator: the optimizer for the discriminator
        identity_loss: loss function used to make the generator which outputs
            a given domain behave as an identity function when given data from
            that domain as input
        cycle_loss: loss function used on the difference between a round-trip
            of the CycleGAN network and the original input
        gan_loss: loss function used on output of the discriminator when
            training the discriminator identify samples correctly or when training
            the generator to fool the discriminator
        batch_size: the number of samples to use in each batch when training
        identity_weight: weight of the identity loss
        cycle_weight: weight of the cycle loss
        generator_weight: weight of the generator's gan loss
        discriminator_weight: weight of the discriminator gan loss
    """

    # This class based loosely on
    # https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/cycle_gan_model.py

    # Copyright Facebook, BSD license
    # https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/c99ce7c4e781712e0252c6127ad1a4e8021cc489/LICENSE

    generator: GeneratorConfig
    optimizer_generator: torch.optim.Optimizer
    optimizer_discriminator: torch.optim.Optimizer
    identity_loss: torch.nn.Module
    cycle_loss: torch.nn.Module
    gan_loss: torch.nn.Module
    batch_size: int
    identity_weight: float = 0.5
    cycle_weight: float = 1.0
    generator_weight: float = 1.0
    discriminator_weight: float = 1.0

    def __post_init__(self):
        self.target_real: Optional[torch.autograd.Variable] = None
        self.target_fake: Optional[torch.autograd.Variable] = None
        # image pool size of 50 used by Zhu et al. (2017)
        self.fake_a_buffer = ImagePool(50)
        self.fake_b_buffer = ImagePool(50)
        self.generator_a_to_b = self.cycle_gan.generator_a_to_b
        self.generator_b_to_a = self.cycle_gan.generator_b_to_a
        self.discriminator_a = self.cycle_gan.discriminator_a
        self.discriminator_b = self.cycle_gan.discriminator_b

    def _init_targets(self, shape: Tuple[int, ...]):
        self.target_real = torch.autograd.Variable(
            torch.Tensor(shape).fill_(1.0).to(DEVICE), requires_grad=False
        )
        self.target_fake = torch.autograd.Variable(
            torch.Tensor(shape).fill_(0.0).to(DEVICE), requires_grad=False
        )

    def evaluate_on_dataset(
        self, dataset: tf.data.Dataset, n_dims_keep: int = 3
    ) -> Dict[str, float]:
        stats_real_a = StatsCollector(n_dims_keep)
        stats_real_b = StatsCollector(n_dims_keep)
        stats_gen_a = StatsCollector(n_dims_keep)
        stats_gen_b = StatsCollector(n_dims_keep)
        real_a: np.ndarray
        real_b: np.ndarray
        reported_plot = False
        for real_a, real_b in dataset:
            # for now there is no time-evolution-based loss, so we fold the time
            # dimension into the sample dimension
            real_a = real_a.reshape(
                [real_a.shape[0] * real_a.shape[1]] + list(real_a.shape[2:])
            )
            real_b = real_b.reshape(
                [real_b.shape[0] * real_b.shape[1]] + list(real_b.shape[2:])
            )
            stats_real_a.observe(real_a)
            stats_real_b.observe(real_b)
            gen_b: np.ndarray = self.generator_a_to_b(
                torch.as_tensor(real_a).float().to(DEVICE)
            ).detach().cpu().numpy()
            gen_a: np.ndarray = self.generator_b_to_a(
                torch.as_tensor(real_b).float().to(DEVICE)
            ).detach().cpu().numpy()
            stats_gen_a.observe(gen_a)
            stats_gen_b.observe(gen_b)
            if not reported_plot and plt is not None:
                report = {}
                for i_tile in range(6):
                    fig, ax = plt.subplots(2, 2, figsize=(8, 7))
                    im = ax[0, 0].pcolormesh(real_a[0, i_tile, 0, :, :])
                    plt.colorbar(im, ax=ax[0, 0])
                    ax[0, 0].set_title("a_real")
                    im = ax[1, 0].pcolormesh(real_b[0, i_tile, 0, :, :])
                    plt.colorbar(im, ax=ax[1, 0])
                    ax[1, 0].set_title("b_real")
                    im = ax[0, 1].pcolormesh(gen_b[0, i_tile, 0, :, :])
                    plt.colorbar(im, ax=ax[0, 1])
                    ax[0, 1].set_title("b_gen")
                    im = ax[1, 1].pcolormesh(gen_a[0, i_tile, 0, :, :])
                    plt.colorbar(im, ax=ax[1, 1])
                    ax[1, 1].set_title("a_gen")
                    plt.tight_layout()
                    buf = io.BytesIO()
                    plt.savefig(buf, format="png")
                    plt.close()
                    buf.seek(0)
                    report[f"tile_{i_tile}_example"] = wandb.Image(
                        PIL.Image.open(buf), caption=f"Tile {i_tile} Example",
                    )
                wandb.log(report)
                reported_plot = True
        metrics = {
            "r2_mean_b_against_real_a": get_r2(stats_real_a.mean, stats_gen_b.mean),
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

    def train_on_batch(
        self, real_a: torch.Tensor, real_b: torch.Tensor
    ) -> Mapping[str, float]:
        """
        Train the CycleGAN on a batch of data.

        Args:
            real_a: a batch of data from domain A, should have shape
                [sample, time, tile, channel, y, x]
            real_b: a batch of data from domain B, should have shape
                [sample, time, tile, channel, y, x]
        """
        # for now there is no time-evolution-based loss, so we fold the time
        # dimension into the sample dimension
        real_a = real_a.reshape(
            [real_a.shape[0] * real_a.shape[1]] + list(real_a.shape[2:])
        )
        real_b = real_b.reshape(
            [real_b.shape[0] * real_b.shape[1]] + list(real_b.shape[2:])
        )

        fake_b = self.generator_a_to_b(real_a)
        fake_a = self.generator_b_to_a(real_b)
        reconstructed_a = self.generator_b_to_a(fake_b)
        reconstructed_b = self.generator_a_to_b(fake_a)

        # Generators A2B and B2A ######

        # don't update discriminators when training generators to fool them
        set_requires_grad(
            [self.discriminator_a, self.discriminator_b], requires_grad=False
        )

        # Identity loss
        # G_A2B(B) should equal B if real B is fed
        same_b = self.generator_a_to_b(real_b)
        loss_identity_b = self.identity_loss(same_b, real_b) * self.identity_weight
        # G_B2A(A) should equal A if real A is fed
        same_a = self.generator_b_to_a(real_a)
        loss_identity_a = self.identity_loss(same_a, real_a) * self.identity_weight
        loss_identity = loss_identity_a + loss_identity_b

        # GAN loss
        pred_fake_b = self.discriminator_b(fake_b)
        if self.target_real is None:
            self._init_targets(pred_fake_b.shape)
        loss_gan_a_to_b = (
            self.gan_loss(pred_fake_b, self.target_real) * self.generator_weight
        )

        pred_fake_a = self.discriminator_a(fake_a)
        loss_gan_b_to_a = (
            self.gan_loss(pred_fake_a, self.target_real) * self.generator_weight
        )
        loss_gan = loss_gan_a_to_b + loss_gan_b_to_a

        # Cycle loss
        loss_cycle_a_b_a = self.cycle_loss(reconstructed_a, real_a) * self.cycle_weight
        loss_cycle_b_a_b = self.cycle_loss(reconstructed_b, real_b) * self.cycle_weight
        loss_cycle = loss_cycle_a_b_a + loss_cycle_b_a_b

        # Total loss
        loss_g: torch.Tensor = (loss_identity + loss_gan + loss_cycle)
        self.optimizer_generator.zero_grad()
        loss_g.backward()
        self.optimizer_generator.step()

        # Discriminators A and B ######

        # do update discriminators when training them to identify samples
        set_requires_grad(
            [self.discriminator_a, self.discriminator_b], requires_grad=True
        )

        # Real loss
        pred_real = self.discriminator_a(real_a)
        loss_d_a_real = (
            self.gan_loss(pred_real, self.target_real) * self.discriminator_weight
        )

        # Fake loss
        fake_a = self.fake_a_buffer.query(fake_a)
        pred_a_fake = self.discriminator_a(fake_a.detach())
        loss_d_a_fake = (
            self.gan_loss(pred_a_fake, self.target_fake) * self.discriminator_weight
        )

        # Real loss
        pred_real = self.discriminator_b(real_b)
        loss_d_b_real = (
            self.gan_loss(pred_real, self.target_real) * self.discriminator_weight
        )

        # Fake loss
        fake_b = self.fake_b_buffer.query(fake_b)
        pred_b_fake = self.discriminator_b(fake_b.detach())
        loss_d_b_fake = (
            self.gan_loss(pred_b_fake, self.target_fake) * self.discriminator_weight
        )

        # Total loss
        loss_d: torch.Tensor = (
            loss_d_b_real + loss_d_b_fake + loss_d_a_real + loss_d_a_fake
        )

        self.optimizer_discriminator.zero_grad()
        loss_d.backward()
        self.optimizer_discriminator.step()

        return {
            "b_to_a_gan_loss": float(loss_gan_b_to_a),
            "a_to_b_gan_loss": float(loss_gan_a_to_b),
            "discriminator_a_loss": float(loss_d_a_fake + loss_d_a_real),
            "discriminator_b_loss": float(loss_d_b_fake + loss_d_b_real),
            "cycle_loss": float(loss_cycle),
            "identity_loss": float(loss_identity),
            "generator_loss": float(loss_g),
            "discriminator_loss": float(loss_d),
            "train_loss": float(loss_g + loss_d),
        }


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


@dataclasses.dataclass
class FMRHyperparameters(Hyperparameters):
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
    training: "FMRTrainingConfig" = dataclasses.field(
        default_factory=lambda: FMRTrainingConfig()
    )

    @property
    def variables(self):
        return tuple(self.state_variables)


@dataclasses.dataclass
class FMRTrainingConfig:
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
        train_model: FMRTrainer,
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
        train_model: FMRTrainer,
        train_data_numpy,
        validation_data: Optional[tf.data.Dataset],
    ):
        for i in range(1, self.n_epoch + 1):
            logger.info("starting epoch %d", i)
            train_losses = []
            for batch_state in train_data_numpy:
                train_losses.append(train_model.train_on_batch(batch_state))
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


@register_training_function("fmr", FMRHyperparameters)
def train_fmr(
    hyperparameters: FMRHyperparameters,
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
