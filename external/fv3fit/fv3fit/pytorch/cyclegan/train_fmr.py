import random
from fv3fit._shared.hyperparameters import Hyperparameters
import dataclasses
from fv3fit._shared.predictor import Reloadable

# from fv3fit.pytorch.cyclegan.discriminator_recurrent import (
#     TimeseriesDiscriminator,
#     TimeseriesDiscriminatorConfig,
# )
from fv3fit.pytorch.cyclegan.discriminator import (
    Discriminator,
    DiscriminatorConfig,
)
from fv3fit.pytorch.cyclegan.generator_recurrent import (
    RecurrentGenerator,
    RecurrentGeneratorConfig,
)
from fv3fit.pytorch.cyclegan.image_pool import ImagePool
from fv3fit.pytorch.cyclegan.modules import (
    FoldFirstDimension,
    halo_convolution,
    # halo_timeseries_convolution,
    single_tile_convolution,
    # single_tile_timeseries_convolution,
)
from fv3fit.pytorch.loss import LossConfig
from fv3fit.pytorch.optimizer import OptimizerConfig
from fv3fit.pytorch.predict import (
    _dump_pytorch,
    _load_pytorch,
    _pack_to_tensor,
    _unpack_tensor,
)
import tensorflow as tf
import torch
from torch import nn
from fv3fit.pytorch.system import DEVICE
import tensorflow_datasets as tfds
from fv3fit.tfdataset import apply_to_mapping, sequence_size

import xarray as xr
from fv3fit._shared import io, register_training_function
from typing import (
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
)
from fv3fit.tfdataset import ensure_nd
from fv3fit.pytorch.graph.train import get_Xy_map_fn as get_Xy_map_fn_single_domain
from fv3fit._shared.scaler import (
    StandardScaler,
    get_standard_scaler_mapping,
    get_mapping_standard_scale_func,
)
import logging
import numpy as np
from .cyclegan_trainer import init_weights

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class FMRNetworkConfig:
    """
    Configuration for building and training a full model replacement network.

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
        reload_path: path to a saved FullModelReplacement model to reload
    """

    generator_optimizer: OptimizerConfig = dataclasses.field(
        default_factory=lambda: OptimizerConfig("Adam")
    )
    discriminator_optimizer: OptimizerConfig = dataclasses.field(
        default_factory=lambda: OptimizerConfig("Adam")
    )
    generator: "RecurrentGeneratorConfig" = dataclasses.field(
        default_factory=lambda: RecurrentGeneratorConfig()
    )
    discriminator: "DiscriminatorConfig" = dataclasses.field(
        default_factory=lambda: DiscriminatorConfig()
    )
    convolution_type: str = "conv2d"
    identity_loss: LossConfig = dataclasses.field(default_factory=LossConfig)
    target_loss: LossConfig = dataclasses.field(default_factory=LossConfig)
    gan_loss: LossConfig = dataclasses.field(default_factory=LossConfig)
    identity_weight: float = 1.0
    target_weight: float = 1.0
    generator_weight: float = 1.0
    discriminator_weight: float = 1.0
    reload_path: Optional[str] = None

    def build(
        self,
        n_state: int,
        nx: int,
        ny: int,
        n_batch: int,
        n_time: int,
        state_variables: Sequence[str],
        scalers: Mapping[str, StandardScaler],
    ) -> "FMRTrainer":
        if self.reload_path is None:
            if self.convolution_type == "conv2d":
                convolution = single_tile_convolution
                # timeseries_convolution = single_tile_timeseries_convolution
            elif self.convolution_type == "halo_conv2d":
                convolution = halo_convolution
                # timeseries_convolution = halo_timeseries_convolution
            else:
                raise ValueError(
                    f"convolution_type {self.convolution_type} not supported"
                )
            generator = self.generator.build(
                n_state, nx=nx, ny=ny, n_time=n_time, convolution=convolution
            ).to(DEVICE)
            discriminator = self.discriminator.build(
                n_state,
                convolution=convolution,
                # timeseries_convolution=timeseries_convolution,
            ).to(DEVICE)
            optimizer_generator = self.generator_optimizer.instance(
                generator.parameters()
            )
            optimizer_discriminator = self.discriminator_optimizer.instance(
                discriminator.parameters()
            )
            init_weights(generator)
            init_weights(discriminator)
            fmr = FullModelReplacement(
                model=FMRModule(generator=generator, discriminator=discriminator),
                scalers=scalers,
                state_variables=state_variables,
            )
        else:
            fmr = FullModelReplacement.load(self.reload_path)
            optimizer_generator = self.generator_optimizer.instance(
                fmr.generator.parameters()
            )
            optimizer_discriminator = self.discriminator_optimizer.instance(
                fmr.discriminator.parameters()
            )
        return FMRTrainer(
            fmr=fmr,
            optimizer_generator=optimizer_generator,
            optimizer_discriminator=optimizer_discriminator,
            identity_loss=self.identity_loss.instance,
            target_loss=self.target_loss.instance,
            gan_loss=self.gan_loss.instance,
            batch_size=n_batch,
            identity_weight=self.identity_weight,
            target_weight=self.target_weight,
            generator_weight=self.generator_weight,
            discriminator_weight=self.discriminator_weight,
        )


class FMRModule(nn.Module):
    """Module to pack generator and discriminator into a single module for saving."""

    def __init__(self, generator: nn.Module, discriminator: nn.Module):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator


@io.register("fmr")
class FullModelReplacement(Reloadable):

    _MODEL_FILENAME = "weight.pt"
    _CONFIG_FILENAME = "config.yaml"
    _SCALERS_FILENAME = "scalers.zip"

    def __init__(
        self,
        model: FMRModule,
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
    def generator(self) -> RecurrentGenerator:
        return self.model.generator

    @property
    def discriminator(self) -> nn.Module:
        return FoldFirstDimension(self.model.discriminator)

    @classmethod
    def load(cls, path: str) -> "FullModelReplacement":
        """Load a serialized model from a directory."""
        return _load_pytorch(cls, path)

    def to(self, device) -> "FullModelReplacement":
        model = self.model.to(device)
        return FullModelReplacement(model, scalers=self.scalers, **self.get_config())

    def dump(self, path: str) -> None:
        _dump_pytorch(self, path)

    def get_config(self):
        return {"state_variables": self.state_variables}

    def pack_to_tensor(self, ds: xr.Dataset, timesteps: int) -> torch.Tensor:
        """
        Packs the dataset into a tensor to be used by the pytorch model.

        Subdivides the dataset evenly into windows
        of size (timesteps + 1) with overlapping start and end points.
        Overlapping the window start and ends is necessary so that every
        timestep (evolution from one time to the next) is included within
        one of the windows.

        Args:
            ds: dataset containing values to pack
            timesteps: number timesteps to include in each window after initial time

        Returns:
            tensor of shape [window, time, tile, x, y, feature]
        """
        tensor = _pack_to_tensor(
            ds=ds,
            timesteps=timesteps,
            state_variables=self.state_variables,
            scalers=self.scalers,
        )
        return tensor.permute([0, 1, 2, 5, 3, 4])

    def unpack_tensor(self, data: torch.Tensor) -> xr.Dataset:
        """
        Unpacks the tensor into a dataset.

        Args:
            data: tensor of shape [window, time, tile, x, y, feature]

        Returns:
            xarray dataset with values of shape [window, time, tile, x, y, feature]
        """
        return _unpack_tensor(
            data.permute([0, 1, 2, 4, 5, 3]),
            varnames=self.state_variables,
            scalers=self.scalers,
            dims=["window", "time", "tile", "x", "y", "z"],
        )

    def step_model(self, state: torch.Tensor, timesteps: int):
        """
        Step the model forward.

        Args:
            state: tensor of shape [sample, tile, x, y, feature]
            timesteps: number of timesteps to predict
        Returns:
            tensor of shape [sample, time, tile, x, y, feature], with time dimension
                having length timesteps + 1 and including the initial state
        """
        with torch.no_grad():
            output = self.generator(state, ntime=timesteps + 1)
        return output

    def predict(self, X: xr.Dataset, timesteps: int) -> Tuple[xr.Dataset, xr.Dataset]:
        """
        Predict an output xarray dataset from an input xarray dataset.

        Note that returned datasets include the initial state of the prediction,
        where by definition the model will have perfect skill.

        Args:
            X: input dataset
            timesteps: number of timesteps to predict

        Returns:
            predicted: predicted timeseries data
            reference: true timeseries data from the input dataset
        """
        tensor = self.pack_to_tensor(X, timesteps=timesteps)
        outputs = self.step_model(tensor[:, 0, :], timesteps=timesteps)
        predicted = self.unpack_tensor(outputs)
        reference = self.unpack_tensor(tensor)
        return predicted, reference


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

    fmr: FullModelReplacement
    optimizer_generator: torch.optim.Optimizer
    optimizer_discriminator: torch.optim.Optimizer
    identity_loss: torch.nn.Module
    target_loss: torch.nn.Module
    gan_loss: torch.nn.Module
    batch_size: int
    target_weight: float = 1.0
    identity_weight: float = 1.0
    generator_weight: float = 1.0
    discriminator_weight: float = 1.0

    def __post_init__(self):
        self.target_real: Optional[torch.autograd.Variable] = None
        self.target_fake: Optional[torch.autograd.Variable] = None
        # image pool size of 50 used by Zhu et al. (2017)
        self.fake_buffer = ImagePool(50)

    def _init_targets(self, shape: Tuple[int, ...]):
        self.target_real = torch.autograd.Variable(
            torch.Tensor(shape).fill_(1.0).to(DEVICE), requires_grad=False
        )
        self.target_fake = torch.autograd.Variable(
            torch.Tensor(shape).fill_(0.0).to(DEVICE), requires_grad=False
        )

    @property
    def generator(self) -> RecurrentGenerator:
        return self.fmr.generator

    @property
    def discriminator(self) -> Discriminator:
        return self.fmr.discriminator

    @property
    def reloadable(self) -> FullModelReplacement:
        return self.fmr

    def evaluate_on_dataset(self, dataset: tf.data.Dataset) -> Dict[str, float]:
        return {}

    def train_on_batch(
        self, real: torch.Tensor, evaluate_only=False
    ) -> Mapping[str, float]:
        """
        Train the CycleGAN on a batch of data.

        Args:
            real: a batch of data from domain A, should have shape
                [sample, time, tile, channel, x, y]
            real_b: a batch of data from domain B, should have shape
                [sample, time, tile, channel, x, y]
        """
        # for now there is no time-evolution-based loss, so we fold the time
        # dimension into the sample dimension
        fake = self.generator(real[:, 0, :])

        # Generator ######

        # don't update discriminators when training generator to fool it
        set_requires_grad([self.discriminator], requires_grad=False)

        # GAN loss
        pred_fake = self.discriminator(fake)
        if self.target_real is None:
            self._init_targets(pred_fake.shape)
        loss_identity = (
            self.identity_loss(real[:, 0], fake[:, 0]) * self.identity_weight
        )
        loss_target = self.target_loss(real[:, 1:], fake[:, 1:]) * self.target_weight
        loss_gan = self.gan_loss(pred_fake, self.target_real) * self.generator_weight
        # Total loss
        loss_g: torch.Tensor = (loss_identity + loss_target + loss_gan)
        if not evaluate_only:
            self.optimizer_generator.zero_grad()
            loss_g.backward()
            self.optimizer_generator.step()

        # Discriminator ######

        # do update discriminator when training it to identify samples
        set_requires_grad([self.discriminator], requires_grad=True)

        # Real loss
        pred_real = self.discriminator(real)
        loss_d_real = (
            self.gan_loss(pred_real, self.target_real) * self.discriminator_weight
        )

        # Fake loss
        fake = self.fake_buffer.query(fake)
        pred_fake = self.discriminator(fake.detach())
        loss_d_fake = (
            self.gan_loss(pred_fake, self.target_fake) * self.discriminator_weight
        )

        # Total loss
        loss_d: torch.Tensor = (loss_d_real + loss_d_fake)

        if not evaluate_only:
            self.optimizer_discriminator.zero_grad()
            loss_d.backward()
            self.optimizer_discriminator.step()

        return {
            "target_loss": float(loss_target),
            "identity_loss": float(loss_identity),
            "gan_loss": float(loss_gan),
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
    network: "FMRNetworkConfig" = dataclasses.field(
        default_factory=lambda: FMRNetworkConfig()
    )
    training: "FMRTrainingConfig" = dataclasses.field(
        default_factory=lambda: FMRTrainingConfig()
    )

    @property
    def variables(self):
        return tuple(self.state_variables)


class Trainer(Protocol):
    def train_on_batch(self, batch: torch.Tensor) -> Mapping[str, float]:
        ...

    def evaluate_on_dataset(self, dataset: tf.data.Dataset) -> Dict[str, float]:
        ...

    @property
    def reloadable(self) -> Reloadable:
        ...


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
        train_model: Trainer,
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
        train_model: Trainer,
        train_data_numpy: tf.data.Dataset,
        validation_data: Optional[tf.data.Dataset],
    ):
        for i in range(1, self.n_epoch + 1):
            logger.info("starting epoch %d", i)
            train_losses = []
            for numpy_state in train_data_numpy:
                batch_state = torch.as_tensor(numpy_state).float().to(DEVICE)
                train_losses.append(train_model.train_on_batch(batch_state))
            train_loss = {
                name: np.mean([data[name] for data in train_losses])
                for name in train_losses[0]
            }
            logger.info("train_loss: %s", train_loss)

            if validation_data is not None:
                val_loss = train_model.evaluate_on_dataset(validation_data)
                logger.info("val_loss %s", val_loss)
            target_loss = train_loss["target_loss"]
            train_model.reloadable.dump(f"model_epoch_{i:04d}_loss_{target_loss:0.4f}")

    def _fit_loop_tensor(
        self,
        train_model: Trainer,
        train_data_numpy: tf.data.Dataset,
        validation_data: Optional[tf.data.Dataset],
    ):
        train_states = []
        for numpy_state in train_data_numpy:
            batch_state = torch.as_tensor(numpy_state).float().to(DEVICE)
            train_states.append(batch_state)
        for i in range(1, self.n_epoch + 1):
            logger.info("starting epoch %d", i)
            train_losses = []
            for batch_state in train_states:
                train_losses.append(train_model.train_on_batch(batch_state))
            random.shuffle(train_states)
            train_loss = {
                name: np.mean([data[name] for data in train_losses])
                for name in train_losses[0]
            }
            logger.info("train_loss: %s", train_loss)

            if validation_data is not None:
                val_loss = train_model.evaluate_on_dataset(validation_data)
                logger.info("val_loss %s", val_loss)


def channels_first(data: tf.Tensor) -> tf.Tensor:
    # [batch, time, tile, x, y, z] -> [batch, time, tile, z, x, y]
    return tf.transpose(data, perm=[0, 1, 2, 5, 3, 4])


@register_training_function("fmr", FMRHyperparameters)
def train_fmr(
    hyperparameters: FMRHyperparameters,
    train_batches: tf.data.Dataset,
    validation_batches: Optional[tf.data.Dataset],
) -> FullModelReplacement:
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

    scalers = get_standard_scaler_mapping(sample_batch)
    mapping_scale_func = get_mapping_standard_scale_func(scalers)

    get_Xy = get_Xy_map_fn_single_domain(
        state_variables=hyperparameters.state_variables,
        n_dims=6,  # [batch, time, tile, x, y, z]
        mapping_scale_func=mapping_scale_func,
    )

    if validation_batches is not None:
        val_state = validation_batches.map(get_Xy)
    else:
        val_state = None

    train_state = train_batches.map(get_Xy)

    sample: tf.Tensor = next(iter(train_state))[0]
    trainer = hyperparameters.network.build(
        nx=sample.shape[-3],
        ny=sample.shape[-2],
        n_state=sample.shape[-1],
        n_time=sample.shape[-5],
        n_batch=hyperparameters.training.samples_per_batch,
        state_variables=hyperparameters.state_variables,
        scalers=scalers,
    )

    # time and tile dimensions aren't being used yet while we're using single-tile
    # convolution without a motion constraint, but they will be used in the future

    # MPS backend has a bug where it doesn't properly read striding information when
    # doing 2d convolutions, so we need to use a channels-first data layout
    # from the get-go and do transformations before and after while in numpy/tf space.
    train_state = train_state.map(channels_first)
    if validation_batches is not None:
        val_state = val_state.map(channels_first)

    # batching from the loader is undone here, so we can do our own batching
    # in fit_loop
    train_state = train_state.unbatch()
    if validation_batches is not None:
        val_state = val_state.unbatch()

    hyperparameters.training.fit_loop(
        train_model=trainer, train_data=train_state, validation_data=val_state,
    )
    return trainer.reloadable
