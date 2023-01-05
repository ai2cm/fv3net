from typing import List, Literal, Mapping, Tuple, Optional
from fv3fit._shared.scaler import StandardScaler
from .reloadable import CycleGAN, CycleGANModule
import torch
from .generator import GeneratorConfig
from .discriminator import DiscriminatorConfig
from .modules import single_tile_convolution, halo_convolution
import dataclasses
from fv3fit.pytorch.loss import LossConfig
from fv3fit.pytorch.optimizer import OptimizerConfig
from fv3fit.pytorch.system import DEVICE
import itertools
from .image_pool import ImagePool
import numpy as np

import logging

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class CycleGANNetworkConfig:
    """
    Configuration for building and training a CycleGAN network.

    Attributes:
        generator_optimizer: configuration for the optimizer used to train the
            generator
        discriminator_optimizer: configuration for the optimizer used to train the
            discriminator
        generator: configuration for building the generator network
        discriminator: configuration for building the discriminator network
        identity_loss: loss function used to make the generator which outputs
            a given domain behave as an identity function when given data from
            that domain as input
        cycle_loss: loss function used on the difference between a round-trip
            of the CycleGAN network and the original input
        gan_loss: loss function used on output of the discriminator when
            training the discriminator identify samples correctly or when training
            the generator to fool the discriminator
        identity_weight: weight of the identity loss
        cycle_weight: weight of the cycle loss
        generator_weight: weight of the generator's gan loss
        discriminator_weight: weight of the discriminator gan loss
    """

    generator_optimizer: OptimizerConfig = dataclasses.field(
        default_factory=lambda: OptimizerConfig("Adam")
    )
    discriminator_optimizer: OptimizerConfig = dataclasses.field(
        default_factory=lambda: OptimizerConfig("Adam")
    )
    generator: "GeneratorConfig" = dataclasses.field(
        default_factory=lambda: GeneratorConfig()
    )
    discriminator: "DiscriminatorConfig" = dataclasses.field(
        default_factory=lambda: DiscriminatorConfig()
    )
    convolution_type: str = "conv2d"
    identity_loss: LossConfig = dataclasses.field(default_factory=LossConfig)
    cycle_loss: LossConfig = dataclasses.field(default_factory=LossConfig)
    gan_loss: LossConfig = dataclasses.field(default_factory=LossConfig)
    identity_weight: float = 0.5
    cycle_weight: float = 1.0
    generator_weight: float = 1.0
    discriminator_weight: float = 1.0
    reload_path: Optional[str] = None

    def build(
        self, n_state: int, nx: int, ny: int, n_batch: int, state_variables, scalers
    ) -> "CycleGANTrainer":
        if self.convolution_type == "conv2d":
            convolution = single_tile_convolution
        elif self.convolution_type == "halo_conv2d":
            convolution = halo_convolution
        else:
            raise ValueError(f"convolution_type {self.convolution_type} not supported")
        generator_a_to_b = self.generator.build(
            n_state, nx=nx, ny=ny, convolution=convolution
        )
        generator_b_to_a = self.generator.build(
            n_state, nx=nx, ny=ny, convolution=convolution
        )
        discriminator_a = self.discriminator.build(n_state, convolution=convolution)
        discriminator_b = self.discriminator.build(n_state, convolution=convolution)
        optimizer_generator = self.generator_optimizer.instance(
            itertools.chain(
                generator_a_to_b.parameters(), generator_b_to_a.parameters()
            )
        )
        optimizer_discriminator = self.discriminator_optimizer.instance(
            itertools.chain(discriminator_a.parameters(), discriminator_b.parameters())
        )
        model = CycleGANModule(
            generator_a_to_b=generator_a_to_b,
            generator_b_to_a=generator_b_to_a,
            discriminator_a=discriminator_a,
            discriminator_b=discriminator_b,
        ).to(DEVICE)
        if self.reload_path is not None:
            reloaded = CycleGAN.load(self.reload_path)
            merged_scalers = reloaded.scalers
            model.load_state_dict(reloaded.model.state_dict(), strict=True)
        else:
            init_weights(model)
            merged_scalers = _merge_scaler_mappings(scalers)
        return CycleGANTrainer(
            cycle_gan=CycleGAN(
                model=model, state_variables=state_variables, scalers=merged_scalers,
            ),
            optimizer_generator=optimizer_generator,
            optimizer_discriminator=optimizer_discriminator,
            identity_loss=self.identity_loss.instance,
            cycle_loss=self.cycle_loss.instance,
            gan_loss=self.gan_loss.instance,
            batch_size=n_batch,
            identity_weight=self.identity_weight,
            cycle_weight=self.cycle_weight,
            generator_weight=self.generator_weight,
            discriminator_weight=self.discriminator_weight,
        )


def init_weights(
    net: torch.nn.Module,
    init_type: Literal["normal", "xavier", "kaiming", "orthogonal"] = "normal",
    init_gain: float = 0.02,
):
    """Initialize network weights.

    Args:
        net: network to be initialized
        init_type: the name of an initialization method
        init_gain: scaling factor for normal, xavier and orthogonal.

    Note: We use 'normal' in the original pix2pix and CycleGAN paper.
    But xavier and kaiming might work better for some applications.
    Feel free to try yourself.
    """

    def init_func(module):  # define the initialization function
        classname = module.__class__.__name__
        if hasattr(module, "weight") and classname == "Conv2d":
            if init_type == "normal":
                torch.nn.init.normal_(module.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                torch.nn.init.xavier_normal_(module.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                torch.nn.init.kaiming_normal_(module.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                torch.nn.init.orthogonal_(module.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    "initialization method [%s] is not implemented" % init_type
                )
            if hasattr(module, "bias") and module.bias is not None:
                torch.nn.init.constant_(module.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            # BatchNorm Layer's weight is not a matrix;
            # only normal distribution applies.
            torch.nn.init.normal_(module.weight.data, 1.0, init_gain)
            torch.nn.init.constant_(module.bias.data, 0.0)

    logger.info("initializing network with %s" % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def _merge_scaler_mappings(
    scaler_tuple: Tuple[Mapping[str, StandardScaler], Mapping[str, StandardScaler]]
) -> Mapping[str, StandardScaler]:
    scalers = {}
    for prefix, scaler_map in zip(("a_", "b_"), scaler_tuple):
        for key, scaler in scaler_map.items():
            scalers[prefix + key] = scaler
    return scalers


class StatsCollector:
    """
    Object to track the mean and standard deviation of sampled arrays.
    """

    def __init__(self, n_dims_keep: int):
        self.n_dims_keep = n_dims_keep
        self._sum = 0.0
        self._sum_squared = 0.0
        self._count = 0

    def observe(self, data: np.ndarray):
        """
        Add a new sample to the statistics.
        """
        mean_dims = tuple(range(0, len(data.shape) - self.n_dims_keep))
        data = data.astype(np.float64)
        self._sum += data.mean(axis=mean_dims)
        self._sum_squared += (data ** 2).mean(axis=mean_dims)
        self._count += 1

    @property
    def mean(self) -> np.ndarray:
        """
        Mean of the observed samples.
        """
        return self._sum / self._count

    @property
    def std(self) -> np.ndarray:
        """
        Standard deviation of the observed samples.
        """
        return np.sqrt(self._sum_squared / self._count - self.mean ** 2)


def get_r2(predicted, target) -> float:
    """
    Compute the R^2 statistic for the predicted and target data.
    """
    return 1.0 - np.var(predicted - target) / np.var(target)


@dataclasses.dataclass
class CycleGANTrainer:
    """
    A trainer for a CycleGAN model.

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

    cycle_gan: CycleGAN
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
        self._script_gen_a_to_b = None
        self._script_gen_b_to_a = None
        self._script_disc_a = None
        self._script_disc_b = None

    def _call_generator_a_to_b(self, input):
        if self._script_gen_a_to_b is None:
            self._script_gen_a_to_b = torch.jit.trace(
                self.generator_a_to_b.forward, input
            )
        return self._script_gen_a_to_b(input)

    def _call_generator_b_to_a(self, input):
        if self._script_gen_b_to_a is None:
            self._script_gen_b_to_a = torch.jit.trace(
                self.generator_b_to_a.forward, input
            )
        return self._script_gen_b_to_a(input)

    def _call_discriminator_a(self, input):
        if self._script_disc_a is None:
            self._script_disc_a = torch.jit.trace(self.discriminator_a.forward, input)
        return self._script_disc_a(input)

    def _call_discriminator_b(self, input):
        if self._script_disc_b is None:
            self._script_disc_b = torch.jit.trace(self.discriminator_b.forward, input)
        return self._script_disc_b(input)

    def _init_targets(self, shape: Tuple[int, ...]):
        self.target_real = torch.autograd.Variable(
            torch.Tensor(shape).fill_(1.0).to(DEVICE), requires_grad=False
        )
        self.target_fake = torch.autograd.Variable(
            torch.Tensor(shape).fill_(0.0).to(DEVICE), requires_grad=False
        )

    def train_on_batch(
        self, real_a: torch.Tensor, real_b: torch.Tensor, training: bool = True
    ) -> Mapping[str, float]:
        """
        Train the CycleGAN on a batch of data.

        Args:
            real_a: a batch of data from domain A, should have shape
                [sample, time, tile, channel, y, x]
            real_b: a batch of data from domain B, should have shape
                [sample, time, tile, channel, y, x]
            training: if True, the model will be trained, otherwise we will
                only evaluate the loss.
        """
        # for now there is no time-evolution-based loss, so we fold the time
        # dimension into the sample dimension
        real_a = real_a.reshape(
            [real_a.shape[0] * real_a.shape[1]] + list(real_a.shape[2:])
        )
        real_b = real_b.reshape(
            [real_b.shape[0] * real_b.shape[1]] + list(real_b.shape[2:])
        )

        fake_b = self._call_generator_a_to_b(real_a)
        fake_a = self._call_generator_b_to_a(real_b)
        reconstructed_a = self._call_generator_b_to_a(fake_b)
        reconstructed_b = self._call_generator_a_to_b(fake_a)

        # Generators A2B and B2A ######

        if training:
            # don't update discriminators when training generators to fool them
            set_requires_grad(
                [self.discriminator_a, self.discriminator_b], requires_grad=False
            )

        # Identity loss
        # G_A2B(B) should equal B if real B is fed
        # same_b = self.generator_a_to_b(real_b)
        same_b = self._call_generator_a_to_b(real_b)
        loss_identity_b = self.identity_loss(same_b, real_b) * self.identity_weight
        # G_B2A(A) should equal A if real A is fed
        same_a = self._call_generator_b_to_a(real_a)
        loss_identity_a = self.identity_loss(same_a, real_a) * self.identity_weight
        loss_identity = loss_identity_a + loss_identity_b

        # GAN loss
        pred_fake_b = self._call_discriminator_b(fake_b)
        if self.target_real is None:
            self._init_targets(pred_fake_b.shape)
        loss_gan_a_to_b = (
            self.gan_loss(pred_fake_b, self.target_real) * self.generator_weight
        )

        pred_fake_a = self._call_discriminator_a(fake_a)
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
        if training:
            self.optimizer_generator.zero_grad()
            loss_g.backward()
            self.optimizer_generator.step()

        # Discriminators A and B ######

        if training:
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
        if training:
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
        if training:
            fake_b = self.fake_b_buffer.query(fake_b)
        pred_b_fake = self.discriminator_b(fake_b.detach())
        loss_d_b_fake = (
            self.gan_loss(pred_b_fake, self.target_fake) * self.discriminator_weight
        )

        # Total loss
        loss_d: torch.Tensor = (
            loss_d_b_real + loss_d_b_fake + loss_d_a_real + loss_d_a_fake
        )

        if training:
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
            "regularization_loss": float(loss_cycle + loss_identity),
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
