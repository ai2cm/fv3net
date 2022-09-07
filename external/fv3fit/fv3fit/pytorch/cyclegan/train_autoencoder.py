from fv3fit._shared.hyperparameters import Hyperparameters
import dataclasses
from fv3fit.pytorch.system import DEVICE
import tensorflow as tf
from fv3fit.pytorch.predict import PytorchPredictor
from fv3fit.pytorch.loss import LossConfig
from fv3fit.pytorch.optimizer import OptimizerConfig
from fv3fit.pytorch.training_loop import TrainingConfig

from fv3fit._shared import register_training_function
from typing import (
    List,
    Optional,
    Tuple,
)
from .generator import Generator, GeneratorConfig
from fv3fit.pytorch.graph.train import get_Xy_map_fn
from fv3fit._shared.scaler import (
    get_standard_scaler_mapping,
    get_mapping_standard_scale_func,
)
from toolz import curry
import logging

logger = logging.getLogger(__name__)


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
    training_loop: "TrainingConfig" = dataclasses.field(
        default_factory=lambda: TrainingConfig()
    )
    loss: LossConfig = LossConfig(loss_type="mse")
    noise_amount: float = 0.5

    @property
    def variables(self):
        return tuple(self.state_variables)


@curry
def define_noisy_input(data: tf.Tensor, stdev=0.1) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Given data, return a tuple with a noisy version of the data and the original data.
    """
    noisy = data + tf.random.normal(shape=tf.shape(data), stddev=stdev)
    return (noisy, data)


def flatten_dims(dataset: tf.data.Dataset) -> tf.data.Dataset:
    """Transform [batch, time, tile, x, y, z] to [sample, tile, x, y, z]"""
    return dataset.unbatch().unbatch()


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
    sample_batch = next(
        iter(train_batches.unbatch().batch(hyperparameters.normalization_fit_samples))
    )

    scalers = get_standard_scaler_mapping(sample_batch)
    mapping_scale_func = get_mapping_standard_scale_func(scalers)

    get_state = get_Xy_map_fn(
        state_variables=hyperparameters.state_variables,
        n_dims=6,  # [batch, time, tile, x, y, z]
        mapping_scale_func=mapping_scale_func,
    )

    if validation_batches is not None:
        val_state = validation_batches.map(get_state)
    else:
        val_state = None

    train_state = train_batches.map(get_state)

    train_model = build_model(
        hyperparameters.generator, n_state=next(iter(train_state)).shape[-1]
    )

    logging.debug("training with model structure: %s", train_model)

    optimizer = hyperparameters.optimizer_config

    train_state = flatten_dims(
        train_state.map(channels_first).map(
            define_noisy_input(stdev=hyperparameters.noise_amount)
        )
    )
    if validation_batches is not None:
        val_state = flatten_dims(
            val_state.map(channels_first).map(
                define_noisy_input(stdev=hyperparameters.noise_amount)
            )
        )

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


def channels_first(data: tf.Tensor) -> tf.Tensor:
    return tf.transpose(data, perm=[0, 1, 2, 5, 3, 4])


def build_model(config: GeneratorConfig, n_state: int) -> Generator:
    return config.build(channels=n_state).to(DEVICE)
