import os
import random
from fv3fit import wandb
from fv3fit._shared.hyperparameters import Hyperparameters
import dataclasses
import tensorflow as tf
import torch
from fv3fit.pytorch.system import DEVICE
import tensorflow_datasets as tfds
from fv3fit.tfdataset import (
    apply_to_mapping_with_exclude,
    select_keys,
    apply_to_tuple,
)
from fv3fit._shared import io
import secrets
from datetime import datetime

from fv3fit._shared import register_training_function
from fv3fit._shared.scaler import StandardScaler
from typing import (
    Callable,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    NewType,
    cast,
)

from fv3fit.tfdataset import ensure_nd
import logging
import numpy as np
from .reloadable import CycleGAN
from .cyclegan_trainer import (
    CycleGANNetworkConfig,
    CycleGANTrainer,
    ResultsAggregator,
    unmerge_scaler_mappings,
)
from .reporter import Reporter
from ..optimizer import SchedulerConfig

logger = logging.getLogger(__name__)

DomainSample = NewType("DomainSample", Tuple[torch.Tensor, torch.Tensor])


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
        reload_path: path to a directory containing a saved CycleGAN model to use
            as a starting point for training
    """

    state_variables: List[str]
    normalization_fit_samples: int = 50_000
    network: "CycleGANNetworkConfig" = dataclasses.field(
        default_factory=lambda: CycleGANNetworkConfig()
    )
    training: "CycleGANTrainingConfig" = dataclasses.field(
        default_factory=lambda: CycleGANTrainingConfig()
    )
    reload_path: Optional[str] = None

    @property
    def variables(self):
        return tuple(self.state_variables) + ("time",)


@dataclasses.dataclass
class CycleGANTrainingConfig:
    """
    Attributes:
        n_epoch: number of epochs to train for
        shuffle_buffer_size: number of samples to use for shuffling the training data
        samples_per_batch: number of samples to use per batch
        in_memory: if True, load the entire dataset into memory as pytorch tensors
            before training. Batches will be statically defined but will be shuffled
            between epochs.
        histogram_vmax: maximum value for histograms of model outputs
        checkpoint_path: if given, model checkpoints will be saved to this directory
            marked by timestamp, epoch, and a randomly generated run label
        scheduler: configuration for the scheduler used to adjust the
            learning rate of the optimizer
    """

    n_epoch: int = 20
    shuffle_buffer_size: int = 10
    samples_per_batch: int = 1
    in_memory: bool = False
    histogram_vmax: float = 100.0
    checkpoint_path: Optional[str] = None
    scheduler: SchedulerConfig = dataclasses.field(
        default_factory=lambda: SchedulerConfig(None)
    )

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
            validation_data = validation_data.batch(self.samples_per_batch)
            validation_data = tfds.as_numpy(validation_data)
        if self.in_memory:
            train_states: Iterable[
                Tuple[DomainSample, DomainSample]
            ] = dataset_to_tuples(train_data_numpy)
            if validation_data is not None:
                val_states: Optional[
                    Iterable[Tuple[DomainSample, DomainSample]]
                ] = dataset_to_tuples(validation_data)
            else:
                val_states = None
        else:
            train_states = DatasetStateIterator(train_data_numpy)
            if validation_data is not None:
                val_states = DatasetStateIterator(validation_data)
            else:
                val_states = None
        self._fit_loop(train_model, train_states, val_states)

    def _fit_loop(
        self,
        train_model: CycleGANTrainer,
        train_states: Iterable[Tuple[DomainSample, DomainSample]],
        validation_states: Optional[Iterable[Tuple[DomainSample, DomainSample]]],
    ):
        reporter = Reporter()
        for state_a, state_b in train_states:
            train_example_a, train_example_b = (
                (state_a[0], state_a[1][:1, :]),
                (state_b[0], state_b[1][:1, :]),
            )
            break
        if validation_states is not None:
            for state_a, state_b in validation_states:
                val_example_a, val_example_b = (
                    (state_a[0], state_a[1][:1, :]),
                    (state_b[0], state_b[1][:1, :]),
                )
                break
        else:
            val_example_a, val_example_b = None, None  # type: ignore
        # current time as e.g. 20230113-163005
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_label = f"{timestamp}-{secrets.token_hex(4)}"
        if self.checkpoint_path is not None:
            logger.info(
                "Saving checkpoints under %s",
                os.path.join(self.checkpoint_path, f"{run_label}-epoch_###"),
            )
        generator_scheduler = self.scheduler.instance(train_model.optimizer_generator)
        discriminator_scheduler = self.scheduler.instance(
            train_model.optimizer_discriminator
        )
        for i in range(1, self.n_epoch + 1):
            logger.info("starting epoch %d", i)
            train_losses = []
            results_aggregator = ResultsAggregator(histogram_vmax=self.histogram_vmax)
            for state_a, state_b in train_states:
                train_losses.append(
                    train_model.train_on_batch(
                        state_a, state_b, aggregator=results_aggregator
                    )
                )
            if isinstance(train_states, list):
                random.shuffle(train_states)
            train_loss = {
                name: np.mean([data[name] for data in train_losses])
                for name in train_losses[0]
            }
            logger.info("train_loss: %s", train_loss)
            reporter.log(train_loss)
            train_plots = train_model.generate_plots(
                train_example_a, train_example_b, results_aggregator
            )
            reporter.log(train_plots)

            if validation_states is not None:
                val_aggregator = ResultsAggregator(histogram_vmax=self.histogram_vmax)
                val_losses = []
                for state_a, state_b in validation_states:
                    with torch.no_grad():
                        val_losses.append(
                            train_model.train_on_batch(
                                state_a,
                                state_b,
                                training=False,
                                aggregator=val_aggregator,
                            )
                        )
                val_loss = {
                    f"val_{name}": np.mean([data[name] for data in val_losses])
                    for name in val_losses[0]
                }
                reporter.log(val_loss)
                val_plots = train_model.generate_plots(
                    val_example_a, val_example_b, val_aggregator
                )
                reporter.log({f"val_{name}": plot for name, plot in val_plots.items()})
                logger.info("val_loss %s", val_loss)
            wandb.log(reporter.metrics)
            reporter.clear()
            generator_scheduler.step()
            discriminator_scheduler.step()

            if self.checkpoint_path is not None:
                current_path = os.path.join(
                    self.checkpoint_path, f"{run_label}-epoch_{i:03d}"
                )
                io.dump(train_model.cycle_gan, str(current_path))


def dataset_to_tuples(dataset) -> List[Tuple[DomainSample, DomainSample]]:
    states = []
    batch_state: Tuple[np.ndarray, np.ndarray]
    for batch_state in dataset:
        time_a = torch.as_tensor(batch_state[0][0]).float().to(DEVICE)
        state_a = torch.as_tensor(batch_state[0][1]).float().to(DEVICE)
        time_b = torch.as_tensor(batch_state[1][0]).float().to(DEVICE)
        state_b = torch.as_tensor(batch_state[1][1]).float().to(DEVICE)
        tuple_a = cast(DomainSample, (time_a, state_a))
        tuple_b = cast(DomainSample, (time_b, state_b))
        states.append((tuple_a, tuple_b))
    return states


class DatasetStateIterator:
    """Iterator over a dataset that returns states as numpy arrays"""

    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        for batch_state in self.dataset:
            time_a = torch.as_tensor(batch_state[0][0]).float().to(DEVICE)
            state_a = torch.as_tensor(batch_state[0][1]).float().to(DEVICE)
            time_b = torch.as_tensor(batch_state[1][0]).float().to(DEVICE)
            state_b = torch.as_tensor(batch_state[1][1]).float().to(DEVICE)
            yield (time_a, state_a), (time_b, state_b)


def apply_to_tuple_mapping(func, exclude: Optional[Sequence[str]] = None):
    # not sure why, but tensorflow doesn't like parsing
    # apply_to_tuple(apply_to_mapping(func)), so we do it manually
    def wrapped(*tuple_of_mapping):
        return tuple(
            {
                name: func(value) if name not in exclude else value
                for name, value in mapping.items()
            }
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
    """
    Args:
        state_variables: names of state variables to extract
        n_dims: number of dimensions in the state variables
        mapping_scale_funcs: one mapping scale function for each domain for which
            we are generating a state. In practice for a basic CycleGAN this will
            contain two scale functions, one for domain A and one for domain B.
    """
    funcs = tuple(
        get_Xy_map_fn_single_domain(
            state_variables=state_variables, n_dims=n_dims, mapping_scale_func=func
        )
        for func in mapping_scale_funcs
    )

    def Xy_map_fn(*data: Mapping[str, np.ndarray]):
        return tuple(func(entry) for func, entry in zip(funcs, data))

    return Xy_map_fn


def get_Xy_map_fn_single_domain(
    state_variables: Sequence[str],
    n_dims: int,
    mapping_scale_func: Callable[[Mapping[str, np.ndarray]], Mapping[str, np.ndarray]],
):
    """
    Returns a function which when given a tf.data.Dataset with mappings from
    variable name to samples
    returns a tf.data.Dataset whose entries are 2-tuples conntaining
    "time" and tensors of the requested state variables concatenated along
    the feature dimension.

    Args:
        state_variables: names of variables to include in returned tensor
        n_dims: number of dimensions of each sample, including feature dimension
        mapping_scale_func: function which scales data stored as a mapping
            from variable name to array
        data: tf.data.Dataset with mappings from variable name
            to sample tensors

    Returns:
        tf.data.Dataset where each sample is a single tensor
            containing normalized and concatenated state variables
    """
    ensure_dims = apply_to_mapping_with_exclude(ensure_nd(n_dims), exclude=("time",))

    def map_fn(data):
        time = data["time"]
        data = mapping_scale_func(data)
        data = ensure_dims(data)
        data = select_keys(state_variables, data)
        data = tf.concat(data, axis=-1)
        return (time, data)

    return map_fn


def channels_first(data: Tuple[tf.Tensor, tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor]:
    # [batch, time, tile, x, y, z] -> [batch, time, tile, z, x, y]
    # first entry of tuple is state label information (time)
    # which we don't want to transpose
    return (data[0], tf.transpose(data[1], perm=[0, 1, 2, 5, 3, 4]))


def force_cudnn_initialization():
    # workaround for https://stackoverflow.com/questions/66588715/runtimeerror-cudnn-error-cudnn-status-not-initialized-using-pytorch  # noqa: E501
    torch.cuda.empty_cache()
    s = 8
    torch.nn.functional.conv2d(
        torch.zeros(s, s, s, s, device=DEVICE), torch.zeros(s, s, s, s, device=DEVICE)
    )


def get_standard_scaler_mapping(
    sample: Mapping[str, np.ndarray]
) -> Mapping[str, StandardScaler]:
    scalers = {}
    for name, array in sample.items():
        if name != "time":
            s = StandardScaler(n_sample_dims=5)
            s.fit(array)
            scalers[name] = s
    return scalers


def get_mapping_standard_scale_func(
    scalers: Mapping[str, StandardScaler]
) -> Callable[[Mapping[str, np.ndarray]], Mapping[str, np.ndarray]]:
    def scale(data: Mapping[str, np.ndarray]):
        output = {**data}
        for name, array in data.items():
            if name != "time":
                output[name] = scalers[name].normalize(array)
        return output

    return scale


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
    force_cudnn_initialization()
    train_batches = train_batches.map(
        apply_to_tuple_mapping(ensure_nd(6), exclude=("time",))
    )
    sample_batch = next(
        iter(train_batches.unbatch().batch(hyperparameters.normalization_fit_samples))
    )

    if hyperparameters.reload_path is not None:
        reloaded = CycleGAN.load(hyperparameters.reload_path)
        merged_scalers = reloaded.scalers
        scalers = unmerge_scaler_mappings(merged_scalers)
    else:
        scalers = tuple(
            get_standard_scaler_mapping(entry) for entry in sample_batch
        )  # type: ignore
    mapping_scale_funcs = tuple(
        get_mapping_standard_scale_func(scaler) for scaler in scalers
    )

    get_Xy = get_Xy_map_fn(
        state_variables=hyperparameters.state_variables,
        n_dims=6,
        mapping_scale_funcs=mapping_scale_funcs,
    )

    if validation_batches is not None:
        val_state = validation_batches.map(get_Xy)
    else:
        val_state = None

    train_state = train_batches.map(get_Xy)

    sample: tf.Tensor = next(iter(train_state))[0][1]  # discard time of first sample
    assert sample.shape[2] == 6  # tile dimension

    train_model = hyperparameters.network.build(
        nx=sample.shape[-3],
        ny=sample.shape[-2],
        n_state=sample.shape[-1],
        n_batch=hyperparameters.training.samples_per_batch,
        state_variables=hyperparameters.state_variables,
        scalers=scalers,
        reload_path=hyperparameters.reload_path,
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
