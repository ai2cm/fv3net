import itertools
from fv3fit._shared.hyperparameters import Hyperparameters
import random
import dataclasses
from fv3fit._shared.predictor import Dumpable
import tensorflow as tf
from fv3fit.pytorch.loss import LossConfig
from fv3fit.pytorch.optimizer import OptimizerConfig
import xarray as xr
import torch
from fv3fit.pytorch.system import DEVICE
import tensorflow_datasets as tfds
from fv3fit.tfdataset import sequence_size
from fv3fit.pytorch.predict import (
    _load_pytorch,
    _dump_pytorch,
    _pack_to_tensor,
    _unpack_tensor,
)

from fv3fit._shared import register_training_function, io, StandardScaler
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)
from fv3fit.tfdataset import ensure_nd
from .network import Discriminator, Generator, GeneratorConfig, DiscriminatorConfig
from fv3fit.pytorch.graph.train import (
    get_scalers,
    get_mapping_scale_func,
    get_Xy_map_fn as get_Xy_map_fn_single_domain,
)
import logging
import numpy as np

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
        train_model: "CycleGANTrainer",
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

            # real_a = torch.as_tensor(batch_state[0]).float().to(DEVICE)
            # real_b = torch.as_tensor(batch_state[1]).float().to(DEVICE)
            # fake_b = train_model.generator_a_to_b(real_a)
            # fake_a = train_model.generator_b_to_a(real_b)
            # reconstructed_a = train_model.generator_b_to_a(fake_b)
            # reconstructed_b = train_model.generator_a_to_b(fake_a)

            # import matplotlib.pyplot as plt

            # fig, ax = plt.subplots(3, 2, figsize=(8, 8))
            # i = 0
            # iz = 0
            # vmin = -1.5
            # vmax = 1.5
            # ax[0, 0].imshow(
            #     real_a[0, :, :, iz].detach().numpy(), vmin=vmin, vmax=vmax
            # )
            # ax[0, 1].imshow(
            #     real_b[0, :, :, iz].detach().numpy(), vmin=vmin, vmax=vmax
            # )
            # ax[1, 0].imshow(
            #     fake_b[0, :, :, iz].detach().numpy(), vmin=vmin, vmax=vmax
            # )
            # ax[1, 1].imshow(
            #     fake_a[0, :, :, iz].detach().numpy(), vmin=vmin, vmax=vmax
            # )
            # ax[2, 0].imshow(
            #     reconstructed_a[0, :, :, iz].detach().numpy(), vmin=vmin, vmax=vmax
            # )
            # ax[2, 1].imshow(
            #     reconstructed_b[0, :, :, iz].detach().numpy(), vmin=vmin, vmax=vmax
            # )
            # ax[0, 0].set_title("real a")
            # ax[0, 1].set_title("real b")
            # ax[1, 0].set_title("output b")
            # ax[1, 1].set_title("output a")
            # ax[2, 0].set_title("reconstructed a")
            # ax[2, 1].set_title("reconstructed b")
            # plt.tight_layout()
            # plt.show()
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

    scalers = tuple(get_scalers(entry) for entry in sample_batch)
    mapping_scale_funcs = tuple(get_mapping_scale_func(scaler) for scaler in scalers)

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
    train_state = train_state.unbatch().unbatch()
    if validation_batches is not None:
        val_state = val_state.unbatch().unbatch()

    hyperparameters.training_loop.fit_loop(
        train_model=train_model, train_data=train_state, validation_data=val_state,
    )
    return train_model.cycle_gan


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
        self._sum = 0.0
        self._sum_squared = 0.0
        self._count = 0

    def observe(self, data: np.ndarray):
        mean_dims = tuple(range(0, len(data.shape) - self.n_dims_keep))
        data = data.astype(np.float64)
        self._sum += data.mean(axis=mean_dims)
        self._sum_squared += (data ** 2).mean(axis=mean_dims)
        self._count += 1

    @property
    def mean(self) -> np.ndarray:
        return self._sum / self._count

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self._sum_squared / self._count - self.mean ** 2)


def get_r2(predicted, target) -> float:
    """
    Compute the R^2 statistic for the predicted and target data.
    """
    return 1.0 - np.var(predicted - target) / np.var(target)


@dataclasses.dataclass
class CycleGANNetworkConfig:
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
    identity_loss: LossConfig = dataclasses.field(default_factory=LossConfig)
    cycle_loss: LossConfig = dataclasses.field(default_factory=LossConfig)
    gan_loss: LossConfig = dataclasses.field(default_factory=LossConfig)
    identity_weight: float = 0.5
    cycle_weight: float = 1.0
    gan_weight: float = 1.0
    discriminator_weight: float = 1.0

    def build(
        self, n_state: int, n_batch: int, state_variables, scalers
    ) -> "CycleGANTrainer":
        generator_a_to_b = self.generator.build(n_state)
        generator_b_to_a = self.generator.build(n_state)
        discriminator_a = self.discriminator.build(n_state)
        discriminator_b = self.discriminator.build(n_state)
        optimizer_generator = self.generator_optimizer.instance(
            itertools.chain(
                generator_a_to_b.parameters(), generator_b_to_a.parameters()
            )
        )
        optimizer_discriminator = self.discriminator_optimizer.instance(
            itertools.chain(discriminator_a.parameters(), discriminator_b.parameters())
        )
        return CycleGANTrainer(
            cycle_gan=CycleGAN(
                model=CycleGANModule(
                    generator_a_to_b=generator_a_to_b,
                    generator_b_to_a=generator_b_to_a,
                    discriminator_a=discriminator_a,
                    discriminator_b=discriminator_b,
                ).to(DEVICE),
                state_variables=state_variables,
                scalers=_merge_scaler_mappings(scalers),
            ),
            optimizer_generator=optimizer_generator,
            optimizer_discriminator=optimizer_discriminator,
            identity_loss=self.identity_loss.instance,
            cycle_loss=self.cycle_loss.instance,
            gan_loss=self.gan_loss.instance,
            batch_size=n_batch,
            identity_weight=self.identity_weight,
            cycle_weight=self.cycle_weight,
            gan_weight=self.gan_weight,
            discriminator_weight=self.discriminator_weight,
        )


def _merge_scaler_mappings(
    scaler_tuple: Tuple[Mapping[str, StandardScaler], Mapping[str, StandardScaler]]
) -> Mapping[str, StandardScaler]:
    scalers = {}
    for prefix, scaler_map in zip(("a_", "b_"), scaler_tuple):
        for key, scaler in scaler_map.items():
            scalers[prefix + key] = scaler
    return scalers


class CycleGANModule(torch.nn.Module):
    def __init__(
        self,
        generator_a_to_b: Generator,
        generator_b_to_a: Generator,
        discriminator_a: Discriminator,
        discriminator_b: Discriminator,
    ):
        super(CycleGANModule, self).__init__()
        self.generator_a_to_b = generator_a_to_b
        self.generator_b_to_a = generator_b_to_a
        self.discriminator_a = discriminator_a
        self.discriminator_b = discriminator_b


@io.register("cycle_gan")
class CycleGAN(Dumpable):

    _MODEL_FILENAME = "weight.pt"
    _CONFIG_FILENAME = "config.yaml"
    _SCALERS_FILENAME = "scalers.zip"

    def __init__(
        self,
        model: CycleGANModule,
        scalers: Mapping[str, StandardScaler],
        state_variables: Iterable[str],
    ):
        """
            Args:
                model: pytorch model
                scalers: scalers for the state variables, keys are prepended with "a_"
                    or "b_" to denote the domain of the scaler, followed by the name of
                    the state variable it scales
                state_variables: name of variables to be used as state variables in
                    the order expected by the model
        """
        self.model = model
        self.scalers = scalers
        self.state_variables = state_variables

    @property
    def generator_a_to_b(self) -> torch.nn.Module:
        return self.model.generator_a_to_b

    @property
    def generator_b_to_a(self) -> torch.nn.Module:
        return self.model.generator_b_to_a

    @property
    def discriminator_a(self) -> torch.nn.Module:
        return self.model.discriminator_a

    @property
    def discriminator_b(self) -> torch.nn.Module:
        return self.model.discriminator_b

    @classmethod
    def load(cls, path: str) -> "CycleGAN":
        """Load a serialized model from a directory."""
        return _load_pytorch(cls, path)

    def dump(self, path: str) -> None:
        _dump_pytorch(self, path)

    def get_config(self):
        return {}

    def pack_to_tensor(self, ds: xr.Dataset, domain: str = "a") -> torch.Tensor:
        """
        Packs the dataset into a tensor to be used by the pytorch model.

        Subdivides the dataset evenly into windows
        of size (timesteps + 1) with overlapping start and end points.
        Overlapping the window start and ends is necessary so that every
        timestep (evolution from one time to the next) is included within
        one of the windows.

        Args:
            ds: dataset containing values to pack
            domain: one of "a" or "b"

        Returns:
            tensor of shape [window, time, tile, x, y, feature]
        """
        scalers = {
            name[2:]: scaler
            for name, scaler in self.scalers.items()
            if name.startswith(f"{domain}_")
        }
        return _pack_to_tensor(
            ds=ds, timesteps=0, state_variables=self.state_variables, scalers=scalers,
        )

    def unpack_tensor(self, data: torch.Tensor, domain: str = "b") -> xr.Dataset:
        """
        Unpacks the tensor into a dataset.

        Args:
            data: tensor of shape [window, time, tile, x, y, feature]
            domain: one of "a" or "b"

        Returns:
            xarray dataset with values of shape [window, time, tile, x, y, feature]
        """
        scalers = {
            name[2:]: scaler
            for name, scaler in self.scalers.items()
            if name.startswith(f"{domain}_")
        }
        return _unpack_tensor(
            data,
            varnames=self.state_variables,
            scalers=scalers,
            dims=["time", "tile", "x", "y", "z"],
        )

    def predict(self, X: xr.Dataset, reverse: bool = False) -> xr.Dataset:
        """
        Predict a state in the output domain from a state in the input domain.

        Args:
            X: input dataset
            reverse: if True, transform from the output domain to the input domain

        Returns:
            predicted: predicted dataset
        """
        if reverse:
            input_domain, output_domain = "b", "a"
        else:
            input_domain, output_domain = "a", "b"

        tensor = self.pack_to_tensor(X, domain=input_domain)
        reshaped_tensor = tensor.reshape(
            [tensor.shape[0] * tensor.shape[1]] + list(tensor.shape[2:])
        )
        with torch.no_grad():
            if reverse:
                outputs = self.generator_b_to_a(reshaped_tensor)
            else:
                outputs = self.generator_a_to_b(reshaped_tensor)
        outputs = outputs.reshape(tensor.shape)
        predicted = self.unpack_tensor(outputs, domain=output_domain)
        return predicted


@dataclasses.dataclass
class CycleGANTrainer:

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
    gan_weight: float = 1.0
    discriminator_weight: float = 1.0

    def __post_init__(self):
        self.target_real = torch.autograd.Variable(
            torch.Tensor(self.batch_size).fill_(1.0).to(DEVICE), requires_grad=False
        )
        self.target_fake = torch.autograd.Variable(
            torch.Tensor(self.batch_size).fill_(0.0).to(DEVICE), requires_grad=False
        )
        self.fake_a_buffer = ReplayBuffer()
        self.fake_b_buffer = ReplayBuffer()
        self.generator_a_to_b = self.cycle_gan.generator_a_to_b
        self.generator_b_to_a = self.cycle_gan.generator_b_to_a
        self.discriminator_a = self.cycle_gan.discriminator_a
        self.discriminator_b = self.cycle_gan.discriminator_b

    def evaluate_on_dataset(
        self, dataset: tf.data.Dataset, n_dims_keep: int = 3
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
            gen_b: torch.Tensor = self.generator_a_to_b(
                torch.as_tensor(real_a).float().to(DEVICE)
            )
            gen_a: torch.Tensor = self.generator_b_to_a(
                torch.as_tensor(real_b).float().to(DEVICE)
            )
            stats_gen_a.observe(gen_a.detach().cpu().numpy())
            stats_gen_b.observe(gen_b.detach().cpu().numpy())
        metrics = {
            # "r2_mean_b_against_real_a": get_r2(stats_real_a.mean, stats_gen_b.mean),
            "r2_mean_a": get_r2(stats_real_a.mean, stats_gen_a.mean),
            # "bias_mean_a": np.mean(stats_real_a.mean - stats_gen_a.mean),
            "r2_mean_b": get_r2(stats_real_b.mean, stats_gen_b.mean),
            # "bias_mean_b": np.mean(stats_real_b.mean - stats_gen_b.mean),
            "r2_std_a": get_r2(stats_real_a.std, stats_gen_a.std),
            # "bias_std_a": np.mean(stats_real_a.std - stats_gen_a.std),
            "r2_std_b": get_r2(stats_real_b.std, stats_gen_b.std),
            # "bias_std_b": np.mean(stats_real_b.std - stats_gen_b.std),
        }
        return metrics

    def train_on_batch(
        self, real_a: torch.Tensor, real_b: torch.Tensor
    ) -> Mapping[str, float]:
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
        loss_gan_a_to_b = self.gan_loss(pred_fake_b, self.target_real) * self.gan_weight

        pred_fake_a = self.discriminator_a(fake_a)
        loss_gan_b_to_a = self.gan_loss(pred_fake_a, self.target_real) * self.gan_weight
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
            self.gan_loss(pred_real, self.target_real)
            * self.gan_weight
            * self.discriminator_weight
        )

        # Fake loss
        fake_a = self.fake_a_buffer.push_and_pop(fake_a)
        pred_a_fake = self.discriminator_a(fake_a.detach())
        loss_d_a_fake = (
            self.gan_loss(pred_a_fake, self.target_fake)
            * self.gan_weight
            * self.discriminator_weight
        )

        # Real loss
        pred_real = self.discriminator_b(real_b)
        loss_d_b_real = (
            self.gan_loss(pred_real, self.target_real)
            * self.gan_weight
            * self.discriminator_weight
        )

        # Fake loss
        fake_b = self.fake_b_buffer.push_and_pop(fake_b)
        pred_b_fake = self.discriminator_b(fake_b.detach())
        loss_d_b_fake = (
            self.gan_loss(pred_b_fake, self.target_fake)
            * self.gan_weight
            * self.discriminator_weight
        )

        # Total loss
        loss_d: torch.Tensor = (
            loss_d_b_real + loss_d_b_fake + loss_d_a_real + loss_d_a_fake
        ) * self.discriminator_weight

        self.optimizer_discriminator.zero_grad()
        loss_d.backward()
        self.optimizer_discriminator.step()

        return {
            # "gan_loss": float(loss_gan),
            "b_to_a_gan_loss": float(loss_gan_b_to_a),
            "a_to_b_gan_loss": float(loss_gan_a_to_b),
            "discriminator_a_loss": float(loss_d_a_fake + loss_d_a_real),
            "discriminator_b_loss": float(loss_d_b_fake + loss_d_b_real),
            # "cycle_loss": float(loss_cycle),
            # "identity_loss": float(loss_identity),
            # "generator_loss": float(loss_g),
            # "discriminator_loss": float(loss_d),
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
