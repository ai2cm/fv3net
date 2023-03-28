from typing import List, Literal, Mapping, Sequence, Tuple, Optional
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
from fv3fit import wandb
import io
import PIL
import xarray as xr
from vcm.cubedsphere import to_cross

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

import logging

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class CycleGANNetworkConfig:
    """
    Configuration for building and training a CycleGAN network.

    Attributes:
        optimizer: configuration for the optimizer used to train the
            generator and discriminator
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

    optimizer: OptimizerConfig = dataclasses.field(
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

    def build(
        self,
        n_state: int,
        nx: int,
        ny: int,
        n_batch: int,
        state_variables: Sequence[str],
        scalers: Tuple[Mapping[str, StandardScaler], Mapping[str, StandardScaler]],
        reload_path: Optional[str] = None,
    ) -> "CycleGANTrainer":
        """
        Build a CycleGANTrainer object.

        Args:
            n_state: number of state variables in the input data
            nx: number of grid points in the x direction
            ny: number of grid points in the y direction
            n_batch: number of samples in a batch
            state_variables: list of state variable names
            scalers: mapping from state variable name to StandardScaler
                for domain A and B
            reload_path: path to a directory containing a saved CycleGAN model to use
                as a starting point for training
        """
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
        discriminator_a = self.discriminator.build(
            n_state, nx=nx, ny=ny, convolution=convolution
        )
        discriminator_b = self.discriminator.build(
            n_state, nx=nx, ny=ny, convolution=convolution
        )
        optimizer_generator = self.optimizer.instance(
            itertools.chain(
                generator_a_to_b.parameters(), generator_b_to_a.parameters()
            )
        )
        optimizer_discriminator = self.optimizer.instance(
            itertools.chain(discriminator_a.parameters(), discriminator_b.parameters())
        )
        model = CycleGANModule(
            generator_a_to_b=generator_a_to_b,
            generator_b_to_a=generator_b_to_a,
            discriminator_a=discriminator_a,
            discriminator_b=discriminator_b,
        ).to(DEVICE)
        if reload_path is not None:
            reloaded = CycleGAN.load(reload_path)
            merged_scalers = reloaded.scalers
            model.load_state_dict(reloaded.model.state_dict(), strict=True)
        else:
            init_weights(model)
            merged_scalers = merge_scaler_mappings(scalers)
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


def merge_scaler_mappings(
    scaler_tuple: Tuple[Mapping[str, StandardScaler], Mapping[str, StandardScaler]]
) -> Mapping[str, StandardScaler]:
    scalers = {}
    for prefix, scaler_map in zip(("a_", "b_"), scaler_tuple):
        for key, scaler in scaler_map.items():
            scalers[prefix + key] = scaler
    return scalers


def unmerge_scaler_mappings(
    scaler_mapping: Mapping[str, StandardScaler],
) -> Tuple[Mapping[str, StandardScaler], Mapping[str, StandardScaler]]:
    """Inverse of merge_scaler_mappings above."""
    scalers_a = {}
    scalers_b = {}
    for key, scaler in scaler_mapping.items():
        if key.startswith("a_"):
            scalers_a[key[2:]] = scaler
        elif key.startswith("b_"):
            scalers_b[key[2:]] = scaler
        else:
            raise ValueError(f"Key {key} does not start with a_ or b_")
    return scalers_a, scalers_b


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


def _hist_multi_channel(array: np.ndarray, bins: np.ndarray):
    """
    Compute a histogram for each channel of a multi-channel array.

    Args:
        array: array of shape (n_samples, n_tiles, n_channels, ...)
        bins: array of shape (n_bins + 1,)

    Returns:
        array of shape (n_channels, n_bins)
    """
    out_hist = np.empty((array.shape[2], bins.shape[0] - 1), dtype=np.float64)
    for i_channel in range(array.shape[2]):
        out_hist[i_channel, :] = np.histogram(
            array[:, :, i_channel].flatten(), bins=bins, density=True
        )[0]
    return out_hist


class ResultsAggregator:
    def __init__(self, histogram_vmax: float):
        self._total_fake_a: Optional[np.ndarray] = None
        self._total_fake_b: Optional[np.ndarray] = None
        self._total_real_a: Optional[np.ndarray] = None
        self._total_real_b: Optional[np.ndarray] = None
        self._total_fake_a_histogram: Optional[np.ndarray] = None
        self._total_fake_b_histogram: Optional[np.ndarray] = None
        self._total_real_a_histogram: Optional[np.ndarray] = None
        self._total_real_b_histogram: Optional[np.ndarray] = None
        self._total_real_a_binned: Optional[np.ndarray] = None
        self._total_real_b_binned: Optional[np.ndarray] = None
        self._total_fake_a_binned: Optional[np.ndarray] = None
        self._total_fake_b_binned: Optional[np.ndarray] = None
        n_bins = 100
        v1 = 10 ** (np.log10(histogram_vmax) / n_bins)
        self._bins = np.concatenate(
            [
                [-1.0, 0.0],
                np.logspace(np.log10(v1), np.log10(histogram_vmax), n_bins - 1),
            ]
        )
        self._count = 0

    def record_results(
        self,
        real_a: np.ndarray,
        real_b: np.ndarray,
        fake_a: np.ndarray,
        fake_b: np.ndarray,
    ):
        """
        Record the results of a single batch.

        Args:
            real_a: Real sample from domain A of shape
                (n_samples, n_tiles, n_channels, height, width).
            real_b: Real sample from domain B of shape
                (n_samples, n_tiles, n_channels, height, width).
            fake_a: Fake sample from domain A of shape
                (n_samples, n_tiles, n_channels, height, width).
            fake_b: Fake sample from domain B of shape
                (n_samples, n_tiles, n_channels, height, width).
        """
        assert len(real_a.shape) == 5
        if self._total_real_a is None:
            self._total_real_a = real_a.mean(axis=0)
        else:
            self._total_real_a += real_a.mean(axis=0)
        if self._total_real_b is None:
            self._total_real_b = real_b.mean(axis=0)
        else:
            self._total_real_b += real_b.mean(axis=0)
        if self._total_fake_b is None:
            self._total_fake_b = fake_b.mean(axis=0)
        else:
            self._total_fake_b += fake_b.mean(axis=0)
        if self._total_fake_a is None:
            self._total_fake_a = fake_a.mean(axis=0)
        else:
            self._total_fake_a += fake_a.mean(axis=0)

        if self._total_real_a_histogram is None:
            self._total_real_a_histogram = _hist_multi_channel(real_a, self._bins)
        else:
            new_hist = _hist_multi_channel(real_a, self._bins)
            self._total_real_a_histogram += new_hist
        if self._total_real_b_histogram is None:
            self._total_real_b_histogram = _hist_multi_channel(real_b, self._bins)
        else:
            new_hist = _hist_multi_channel(real_b, self._bins)
            self._total_real_b_histogram += new_hist
        if self._total_fake_a_histogram is None:
            self._total_fake_a_histogram = _hist_multi_channel(fake_a, self._bins)
        else:
            new_hist = _hist_multi_channel(fake_a, self._bins)
            self._total_fake_a_histogram += new_hist
        if self._total_fake_b_histogram is None:
            self._total_fake_b_histogram = _hist_multi_channel(fake_b, self._bins)
        else:
            new_hist = _hist_multi_channel(fake_b, self._bins)
            self._total_fake_b_histogram += new_hist

        self._count += 1

    @property
    def mean_fake_a(self) -> np.ndarray:
        if self._total_fake_a is None:
            raise RuntimeError("No results have been recorded yet.")
        return self._total_fake_a / self._count

    @property
    def mean_fake_b(self) -> np.ndarray:
        if self._total_fake_b is None:
            raise RuntimeError("No results have been recorded yet.")
        return self._total_fake_b / self._count

    @property
    def mean_real_a(self) -> np.ndarray:
        if self._total_real_a is None:
            raise RuntimeError("No results have been recorded yet.")
        return self._total_real_a / self._count

    @property
    def mean_real_b(self) -> np.ndarray:
        if self._total_real_b is None:
            raise RuntimeError("No results have been recorded yet.")
        return self._total_real_b / self._count

    @property
    def fake_a_histogram(self) -> np.ndarray:
        if self._total_fake_a_histogram is None:
            raise RuntimeError("No results have been recorded yet.")
        return self._total_fake_a_histogram / self._count

    @property
    def fake_b_histogram(self) -> np.ndarray:
        if self._total_fake_b_histogram is None:
            raise RuntimeError("No results have been recorded yet.")
        return self._total_fake_b_histogram / self._count

    @property
    def real_a_histogram(self) -> np.ndarray:
        if self._total_real_a_histogram is None:
            raise RuntimeError("No results have been recorded yet.")
        return self._total_real_a_histogram / self._count

    @property
    def real_b_histogram(self) -> np.ndarray:
        if self._total_real_b_histogram is None:
            raise RuntimeError("No results have been recorded yet.")
        return self._total_real_b_histogram / self._count

    @property
    def bins(self) -> np.ndarray:
        return self._bins


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
        non_negativity_weight: weight of the non-negativity L1 loss
        metric_percentiles: the percentiles to use when computing the
            histogram error metrics
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
    cycle_weight: float = 10.0
    generator_weight: float = 1.0
    discriminator_weight: float = 1.0
    non_negativity_weight: float = 0.0
    metric_percentiles: List[float] = dataclasses.field(
        default_factory=lambda: [0.99, 0.999, 0.9999]
    )

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
        self._l1_loss = torch.nn.L1Loss()
        # This flag can be manually used to disable compilation, for clearer
        # error messages and debugging. It should always be set to True in PRs.
        self._compile = False

    def _call_generator_a_to_b(
        self, input: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        if self._compile:
            if self._script_gen_a_to_b is None:
                self._script_gen_a_to_b = torch.jit.trace(
                    self.generator_a_to_b.forward, input,
                )
            try:
                return self._script_gen_a_to_b(*input)
            except RuntimeError:
                # gives better messages for true errors, but also lets us process
                # a smaller batch at the end of the dataset if one exists
                return self.generator_a_to_b(*input)
        else:
            return self.generator_a_to_b(*input)

    def _call_generator_b_to_a(
        self, input: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        if self._compile:
            if self._script_gen_b_to_a is None:
                self._script_gen_b_to_a = torch.jit.trace(
                    self.generator_b_to_a.forward, input,
                )
            try:
                return self._script_gen_b_to_a(*input)
            except RuntimeError:
                return self.generator_b_to_a(*input)
        else:
            return self.generator_b_to_a(*input)

    def _call_discriminator_a(
        self, input: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        if self._compile:
            if self._script_disc_a is None:
                self._script_disc_a = torch.jit.trace(
                    self.discriminator_a.forward, input,
                )
            try:
                return self._script_disc_a(*input)
            except RuntimeError:
                return self.discriminator_a(*input)
        else:
            return self.discriminator_a(*input)

    def _call_discriminator_b(
        self, input: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        if self._compile:
            if self._script_disc_b is None:
                self._script_disc_b = torch.jit.trace(
                    self.discriminator_b.forward, input,
                )
            try:
                return self._script_disc_b(*input)
            except RuntimeError:
                return self.discriminator_b(*input)
        else:
            return self.discriminator_b(*input)

    def _non_negativity_loss(self, output: torch.Tensor) -> torch.Tensor:
        # we want to penalize the generator for generating negative values
        # so we use the L1 loss
        # to only affect negative values we look at min(0, output)
        zeros = torch.zeros_like(output)
        return self._l1_loss(torch.min(output, zeros), zeros)

    def train_on_batch(
        self,
        state_a: Tuple[torch.Tensor, torch.Tensor],
        state_b: Tuple[torch.Tensor, torch.Tensor],
        training: bool = True,
        aggregator: Optional[ResultsAggregator] = None,
    ) -> Mapping[str, float]:
        """
        Train the CycleGAN on a batch of data.

        Args:
            state_a: a tuple containing a "time" tensor of shape [sample, time]
                and a batch of data from domain A, should have shape
                [sample, time, tile, channel, y, x]
            state_b: a tuple containing a "time" tensor of shape [sample, time]
                and a batch of data from domain B, should have shape
                [sample, time, tile, channel, y, x]
            training: if True, the model will be trained, otherwise we will
                only evaluate the loss.
            aggregator: if given, record generated results in this aggregator
        """
        if aggregator is None:
            aggregator = ResultsAggregator(histogram_vmax=100.0)
        time_a, time_b, real_a, real_b = unpack_state(state_a, state_b)

        fake_b = self._call_generator_a_to_b((time_a, real_a))
        fake_a = self._call_generator_b_to_a((time_b, real_b))
        reconstructed_a = self._call_generator_b_to_a((time_a, fake_b))
        reconstructed_b = self._call_generator_a_to_b((time_b, fake_a))

        # Generators A2B and B2A ######

        if training:
            # don't update discriminators when training generators to fool them
            set_requires_grad(
                [self.discriminator_a, self.discriminator_b], requires_grad=False
            )

        # Identity loss
        # G_A2B(B) should equal B if real B is fed
        # same_b = self.generator_a_to_b(real_b)
        same_b = self._call_generator_a_to_b((time_b, real_b))
        loss_identity_b = self.identity_loss(same_b, real_b) * self.identity_weight
        # G_B2A(A) should equal A if real A is fed
        same_a = self._call_generator_b_to_a((time_a, real_a))
        loss_identity_a = self.identity_loss(same_a, real_a) * self.identity_weight
        loss_identity = loss_identity_a + loss_identity_b

        # GAN loss
        pred_fake_b = self._call_discriminator_b((time_a, fake_b))
        if self.target_real is None:
            self.target_real = torch.autograd.Variable(
                torch.Tensor(pred_fake_b.shape).fill_(1.0).to(DEVICE),
                requires_grad=False,
            )
        if self.target_fake is None:
            self.target_fake = torch.autograd.Variable(
                torch.Tensor(pred_fake_b.shape).fill_(0.0).to(DEVICE),
                requires_grad=False,
            )
        n_samples = pred_fake_b.shape[0]
        # last batch may have fewer samples, so we slice the target output 1/0's
        target_real = self.target_real[:n_samples]
        target_fake = self.target_fake[:n_samples]
        loss_gan_a_to_b = (
            self.gan_loss(pred_fake_b, target_real) * self.generator_weight
        )

        pred_fake_a = self._call_discriminator_a((time_b, fake_a))
        loss_gan_b_to_a = (
            self.gan_loss(pred_fake_a, target_real) * self.generator_weight
        )
        loss_gan = loss_gan_a_to_b + loss_gan_b_to_a

        # Cycle loss
        loss_cycle_a_b_a = self.cycle_loss(reconstructed_a, real_a) * self.cycle_weight
        loss_cycle_b_a_b = self.cycle_loss(reconstructed_b, real_b) * self.cycle_weight
        loss_cycle = loss_cycle_a_b_a + loss_cycle_b_a_b

        # Non-negativity loss
        loss_non_neg_a = self._non_negativity_loss(fake_a) * self.non_negativity_weight
        loss_non_neg_b = self._non_negativity_loss(fake_b) * self.non_negativity_weight
        loss_non_neg = loss_non_neg_a + loss_non_neg_b

        # Total loss
        loss_g: torch.Tensor = (loss_identity + loss_gan + loss_cycle + loss_non_neg)

        with torch.no_grad():
            aggregator.record_results(
                fake_a=fake_a.cpu().numpy(),
                fake_b=fake_b.cpu().numpy(),
                real_a=real_a.cpu().numpy(),
                real_b=real_b.cpu().numpy(),
            )

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
        pred_real = self._call_discriminator_a((time_a, real_a))
        loss_d_a_real = (
            self.gan_loss(pred_real, target_real) * self.discriminator_weight
        )

        # Fake loss
        if training:
            time_b, fake_a = self.fake_a_buffer.query(
                (time_b.detach(), fake_a.detach())
            )
        pred_a_fake = self._call_discriminator_a((time_b, fake_a))
        loss_d_a_fake = (
            self.gan_loss(pred_a_fake, target_fake) * self.discriminator_weight
        )

        # Real loss
        pred_real = self._call_discriminator_b((time_b, real_b))
        loss_d_b_real = (
            self.gan_loss(pred_real, target_real) * self.discriminator_weight
        )

        # Fake loss
        if training:
            time_a, fake_b = self.fake_b_buffer.query(
                (time_a.detach(), fake_b.detach())
            )
        pred_b_fake = self._call_discriminator_b((time_a, fake_b))
        loss_d_b_fake = (
            self.gan_loss(pred_b_fake, target_fake) * self.discriminator_weight
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

    def generate_plots(
        self,
        state_a: Tuple[torch.Tensor, torch.Tensor],
        state_b: Tuple[torch.Tensor, torch.Tensor],
        results_aggregator: Optional[ResultsAggregator] = None,
    ) -> Mapping[str, wandb.Image]:
        """
        Plot model output on the first sample of a given batch and return it as
        a dictionary of wandb.Image objects.

        Args:
            state_a: a tuple containing a "time" tensor of shape [sample, time]
                and a batch of data from domain A, should have shape
                [sample, time, tile, channel, y, x]
            state_b: a tuple containing a "time" tensor of shape [sample, time]
                and a batch of data from domain B, should have shape
                [sample, time, tile, channel, y, x]
            results_aggregator: an aggregator whose results we should plot
        """
        time_a, time_b, real_a, real_b = unpack_state(state_a, state_b)

        # plot the first sample of the batch
        with torch.no_grad():
            fake_b = self._call_generator_a_to_b((time_a[:1], real_a[:1, :]))
            fake_a = self._call_generator_b_to_a((time_b[:1], real_b[:1, :]))
        real_a = real_a.cpu().numpy()
        real_b = real_b.cpu().numpy()
        fake_a = fake_a.cpu().numpy()
        fake_b = fake_b.cpu().numpy()
        report = {}
        for i in range(real_a.shape[2]):
            buf = plot_cross(
                real_a[0, :, i, :, :],
                real_b[0, :, i, :, :],
                fake_a[0, :, i, :, :],
                fake_b[0, :, i, :, :],
            )
            report[f"example_{i}"] = wandb.Image(
                PIL.Image.open(buf), caption=f"Channel {i} Example",
            )

        fig, ax = plt.subplots(
            real_a.shape[2], 2, figsize=(10, 1 + 2.5 * real_a.shape[2])
        )
        if real_a.shape[2] == 1:
            ax = ax[None, :]
        for i in range(real_a.shape[2]):
            plot_hist(
                real_a=real_a[:, :, i, :, :],
                real_b=real_b[:, :, i, :, :],
                gen_a=fake_a[:, :, i, :, :],
                gen_b=fake_b[:, :, i, :, :],
                ax=ax[i, 0],
            )
            plot_hist(
                real_a=real_a[:, :, i, :, :],
                real_b=real_b[:, :, i, :, :],
                gen_a=fake_a[:, :, i, :, :],
                gen_b=fake_b[:, :, i, :, :],
                ax=ax[i, 1],
            )
            ax[i, 1].set_yscale("log")
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        report[f"histogram"] = wandb.Image(PIL.Image.open(buf), caption=f"Histograms",)

        if results_aggregator is not None:
            res = results_aggregator
            # real_a_bias = real_a_mean - real_b_mean
            real_a_bias = res.mean_real_a - res.mean_real_b
            for i in range(real_a.shape[2]):
                # report mean bias
                fake_a_bias = res.mean_fake_a[:, i, :, :] - res.mean_real_a[:, i, :, :]
                fake_b_bias = res.mean_fake_b[:, i, :, :] - res.mean_real_b[:, i, :, :]
                buf = plot_cross(
                    real_a_bias[:, i, :, :],
                    res.mean_real_b[:, i, :, :],
                    fake_a_bias,
                    fake_b_bias,
                    combined_vmin_vmax=False,
                )
                report[f"bias_{i}"] = wandb.Image(
                    PIL.Image.open(buf), caption=f"Channel {i} bias vs real_b",
                )
                report[f"real_a_vs_real_b_bias_mean_{i}"] = np.mean(real_a_bias)
                report[f"real_a_vs_real_b_bias_std_{i}"] = np.std(real_a_bias)
                report[f"fake_a_vs_real_a_bias_mean_{i}"] = np.mean(fake_a_bias)
                report[f"fake_a_vs_real_a_bias_std_{i}"] = np.std(fake_a_bias)
                report[f"fake_b_vs_real_b_bias_mean_{i}"] = np.mean(fake_b_bias)
                report[f"fake_b_vs_real_b_bias_std_{i}"] = np.std(fake_b_bias)
                # report histograms
                buf = plot_histograms(
                    real_a_hist=res.real_a_histogram[i, :],
                    real_b_hist=res.real_b_histogram[i, :],
                    fake_a_hist=res.fake_a_histogram[i, :],
                    fake_b_hist=res.fake_b_histogram[i, :],
                    bins=res.bins,
                )
                report[f"histogram_{i}"] = wandb.Image(
                    PIL.Image.open(buf), caption=f"Channel {i} Histograms",
                )
                # add error in n'th percentile for each percentile
                # in self.metric_percentiles
                for pct in self.metric_percentiles:
                    report[
                        f"percentile_{pct:0.6f}_gen_error_{i}"
                    ] = get_percentile_error(
                        res.bins,
                        res.fake_b_histogram[i, :],
                        res.real_b_histogram[i, :],
                        pct,
                    )
                    report[
                        f"percentile_{pct:0.6f}_a_minus_b_error_{i}"
                    ] = get_percentile_error(
                        res.bins,
                        res.real_a_histogram[i, :],
                        res.real_b_histogram[i, :],
                        pct,
                    )

        return report


def get_percentile(bins: np.ndarray, hist: np.ndarray, pct: float):
    """Returns the pct percentile of the histogram, where pct lies in (0, 1]."""
    # get the normalized CDF based on the histogram
    cdf = np.cumsum(hist)
    cdf = cdf / cdf[-1]
    # append initial zero to cdf as there are no values less than the first bin
    cdf = np.insert(cdf, 0, 0)
    # find within which bin the requested pct percentile falls
    bin_idx = np.argmax(cdf > pct) - 1
    # linearly interpolate within the bin to get the percentile value
    pct_val = bins[bin_idx] + (bins[bin_idx + 1] - bins[bin_idx]) * (
        pct - cdf[bin_idx]
    ) / (cdf[bin_idx + 1] - cdf[bin_idx])
    return pct_val


def get_percentile_error(bins, hist_actual, hist_expected, pct):
    """
    Returns the error in the pct percentile of the histogram,
    where pct lies in (0, 1].
    """
    pct_actual = get_percentile(bins, hist_actual, pct)
    pct_expected = get_percentile(bins, hist_expected, pct)
    return pct_actual - pct_expected


def unpack_state(
    state_a: Tuple[torch.Tensor, torch.Tensor],
    state_b: Tuple[torch.Tensor, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Unpacks states into time and input data, and folds the batch and window dimensions
    into one.
    """

    time_a = state_a[0].flatten()
    time_b = state_b[0].flatten()
    real_a = state_a[1]
    real_b = state_b[1]
    # for now there is no time-evolution-based loss, so we fold the time
    # dimension into the sample dimension
    real_a = real_a.reshape(
        [real_a.shape[0] * real_a.shape[1]] + list(real_a.shape[2:])
    )
    real_b = real_b.reshape(
        [real_b.shape[0] * real_b.shape[1]] + list(real_b.shape[2:])
    )
    return time_a, time_b, real_a, real_b


def plot_histograms(
    real_a_hist: np.ndarray,
    real_b_hist: np.ndarray,
    fake_a_hist: np.ndarray,
    fake_b_hist: np.ndarray,
    bins: np.ndarray,
):
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    ax.step(
        bins[:-1], real_a_hist, where="post", alpha=0.5, label="real_a",
    )
    ax.step(
        bins[:-1], real_b_hist, where="post", alpha=0.5, label="real_b",
    )
    ax.step(
        bins[:-1], fake_a_hist, where="post", alpha=0.5, label="fake_a",
    )
    ax.step(
        bins[:-1], fake_b_hist, where="post", alpha=0.5, label="fake_b",
    )
    i_max = np.max(
        np.where(np.any([real_a_hist, real_b_hist, fake_a_hist, fake_b_hist], axis=0))
    )
    ax.set_xlim(bins[0], bins[i_max + 1])
    ax.set_yscale("log")
    ax.set_title(f"Histograms")
    ax.legend()
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf


def plot_cross(
    real_a: np.ndarray,
    real_b: np.ndarray,
    fake_a: np.ndarray,
    fake_b: np.ndarray,
    combined_vmin_vmax: bool = True,
) -> io.BytesIO:
    """
    Plot global states as cross-plots.

    Args:
        real_a: Real state from domain A, shape [tile, x, y]
        real_b: Real state from domain B, shape [tile, x, y]
        fake_a: Fake state from domain A, shape [tile, x, y]
        fake_b: Fake state from domain B, shape [tile, x, y]

    Returns:
        io.BytesIO: BytesIO object containing the plot
    """

    var_real_a = to_cross(xr.DataArray(real_a, dims=["tile", "grid_xt", "grid_yt"]))
    var_real_b = to_cross(xr.DataArray(real_b, dims=["tile", "grid_xt", "grid_yt"]))
    var_fake_a = to_cross(xr.DataArray(fake_a, dims=["tile", "grid_xt", "grid_yt"]))
    var_fake_b = to_cross(xr.DataArray(fake_b, dims=["tile", "grid_xt", "grid_yt"]))
    if combined_vmin_vmax:
        vmin_a = min(np.min(real_a), np.min(fake_a))
        vmax_a = max(np.max(real_a), np.max(fake_a))
        vmin_b = min(np.min(real_b), np.min(fake_b))
        vmax_b = max(np.max(real_b), np.max(fake_b))
    else:
        vmin_a, vmax_a, vmin_b, vmax_b = None, None, None, None
    fig, ax = plt.subplots(2, 2, figsize=(8, 7))
    var_real_a.plot(ax=ax[0, 0], vmin=vmin_a, vmax=vmax_a)
    var_fake_b.plot(ax=ax[0, 1], vmin=vmin_b, vmax=vmax_b)
    var_real_b.plot(ax=ax[1, 0], vmin=vmin_b, vmax=vmax_b)
    var_fake_a.plot(ax=ax[1, 1], vmin=vmin_a, vmax=vmax_a)
    ax[0, 0].set_title("real_a")
    ax[0, 1].set_title("fake_b")
    ax[1, 0].set_title("real_b")
    ax[1, 1].set_title("fake_a")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf


def plot_hist(real_a, real_b, gen_a, gen_b, ax=None):
    ax.hist(
        real_a.flatten(),
        bins=100,
        alpha=0.5,
        label="real_a",
        histtype="step",
        density=True,
    )
    ax.hist(
        real_b.flatten(),
        bins=100,
        alpha=0.5,
        label="real_b",
        histtype="step",
        density=True,
    )
    ax.hist(
        gen_a.flatten(),
        bins=100,
        alpha=0.5,
        label="gen_a",
        histtype="step",
        density=True,
    )
    ax.hist(
        gen_b.flatten(),
        bins=100,
        alpha=0.5,
        label="gen_b",
        histtype="step",
        density=True,
    )
    ax.legend(loc="upper left")
    ax.set_ylabel("probability density")


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
